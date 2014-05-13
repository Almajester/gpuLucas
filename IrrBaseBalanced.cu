/**
* IrrBaseBalanced.cu -- globals and kernel code to do operations on balanced radix numbers
*   This uses the Crandall irrational base method
*
* A. Thall & A. Hegedus
* Project:  gpuLucas
* 11/6/2010
*
****************************************************************************
*
* Copyright (c) 2010-2012, Andrew Thall, Alma College
* All rights reserved.
*
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
*     * Redistributions of source code must retain the above copyright
*       notice, this list of conditions and the following disclaimer.
*     * Redistributions in binary form must reproduce the above copyright
*       notice, this list of conditions and the following disclaimer in the
*       documentation and/or other materials provided with the distribution.
*     * Neither the names of Andrew Thall or Alma College, nor the
*       names of its contributors may be used to endorse or promote products
*       derived from this software without specific prior written permission.
*
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL ANDREW THALL OR ALMA COLLEGE BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
* ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
* SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
******************************************************************************
*
* MODIFICATIONS:
*  8/1/2011:
*     xxAH:  llintToIrrBal() now template function
*     xxAT:  Removed all dependencies on CUDPP and fast carry addition
*  2/19/2012:  xxAT release version
*/

#ifndef D_IRRBASEBALANCED
#define D_IRRBASEBALANCED

/**
 * Distribute product-accumulated bits to subsequent digits of variable base product
 * @template-param number - number of subsequent digits to distribute product bits
 */
template <int number>
static __global__ void llintToIrrBal(int *iArr, int *hiArr, long long int *lliArr, unsigned char *bperW8arr, const int size) {
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	const int tba = threadIdx.x; // thread block address for digits index

	// Use int for each element, the radix place and its n preceeding
	__shared__ long long int digits[T_PER_B + number];
	__shared__ int signs[T_PER_B + number];
	
	// first n threads of block initialize leading digits.
	//   Be carefule to wrap-around from end of array if tid < n
	//   otherwise, load end of previous block at [tid - n]
	if (tba < number) {
     	int offset;
		if (tid - number < 0) 
			offset = size + tid;
		else
			offset = tid;
		digits[tba] = lliArr[offset - number];
		signs[tba] = digits[tba] < 0 ? -1 : 1;
		digits[tba] *= signs[tba];
	}

	digits[tba + number] = lliArr[tid];
	signs[tba + number] = digits[tba + number] < 0 ? -1 : 1;
	digits[tba + number] *= signs[tba + number]; 
	
	__syncthreads();
	
	unsigned char bperW8 = bperW8arr[tid];

	// get info for this digit
	int isHi = bperW8 & 1;
	int BITS = LO_BITS + isHi;
	int myBase = BASE_LO << isHi;
	int myMask = myBase - 1;

	// Walk backwards through the cached long longs, pulling off
	//   higher and higher order bits, all of length myMask for the
	//   current digit.
	// sbitN is amount to shift word (tid - N) before pulling off
	//   higher order bits with myMask for current digit
	int sbits1, sbits2, sbits3, sbits4, sbits5, sbits6;
	if (number >= 1) 
		sbits1 =          LO_BITS + ((bperW8 >> 1) & 1);
	if (number >= 2) 
		sbits2 = sbits1 + LO_BITS + ((bperW8 >> 2) & 1);
	if (number >= 3) 
		sbits3 = sbits2 + LO_BITS + ((bperW8 >> 3) & 1);
	if (number >= 4) 
		sbits4 = sbits3 + LO_BITS + ((bperW8 >> 4) & 1);
	if (number >= 5) 
		sbits5 = sbits4 + LO_BITS + ((bperW8 >> 5) & 1);
	if (number >= 6) 
		sbits6 = sbits5 + LO_BITS + ((bperW8 >> 6) & 1);

	int sum = signs[tba + number]*(digits[tba + number]              & myMask);
	if(number >= 1)
		sum += signs[tba + number - 1]*((digits[tba + number - 1] >> sbits1) & myMask);
	if(number >= 2)
		sum += signs[tba + number - 2]*((digits[tba + number - 2] >> sbits2) & myMask);
	if(number >= 3)
		sum += signs[tba + number - 3]*((digits[tba + number - 3] >> sbits3) & myMask);
	if(number >= 4)
		sum += signs[tba + number - 4]*((digits[tba + number - 4] >> sbits4) & myMask);
	if(number >= 5)
		sum += signs[tba + number - 5]*((digits[tba + number - 5] >> sbits5) & myMask);
	if(number >= 6)
		sum += signs[tba + number - 6]*((digits[tba + number - 6] >> sbits6) & myMask);

  /* OLD VERSION.  above really doesn't buy much.  below is simpler,
        but not templated.
  	int shiftBits = 0;
	for (int i = 1; i < 6 + 1; i++) {
		bperW8 >>= 1;
		isHi = bperW8 & 1;
		shiftBits += LO_BITS + isHi;
		sum += signs[tba + 6 - i]*((digits[tba + 6 - i] >> shiftBits) &  myMask);
	}
    */

	// do pseudo-rebalance, storing borrow or carry to hiArr[tid]
	int baseOver2 = myBase >> 1;
	int hival = 0;
	if (sum < -baseOver2)
		hival = -((-sum + baseOver2) >> BITS); //  /myBase);
	else if (sum >= baseOver2)
		hival = (sum + baseOver2) >> BITS; // /myBase;

	iArr[tid] = sum - (hival << BITS);
	hiArr[tid] = hival;
}

/**
 * do a single carry of the high-order carry of the previous digit to the
 *    current digit.  Don't rebalance if exceeds max or min on balanced
 *    representation.
 */
static __global__ void addPseudoBalanced(int *signal, int *hiAdd, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid == 0) 
		signal[tid] += hiAdd[size - 1];
	else
		signal[tid] += hiAdd[tid - 1];
}

/**
 * Final rebalance of irrational base representation, by one-time-only sequential
 *   add-with-carry with rebalancedIrrIntSEQGPU<<<1, 1>>> call.  Could as easily be
 *   done CPU-side.
 */
static __global__ void rebalanceIrrIntSEQGPU(int *signal, unsigned char *bpwArr, int size) {

	int carryOut = 0;
	int tBase, tBaseOver2;
	int BASE_HIOVER2 = BASE_HI >> 1;
	int BASE_LOOVER2 = BASE_LO >> 1;

	for (int i = 0; i < size; i++) {

		if (bpwArr[i] & 1) {
			tBase = BASE_HI;
			tBaseOver2 = BASE_HIOVER2;
		}
		else {
			tBase = BASE_LO;
			tBaseOver2 = BASE_LOOVER2;
		}
		int b = signal[i];

		int total = b + carryOut;

		if (total >= tBaseOver2) {
			signal[i] = total - tBase;
			carryOut = 1;
		}
		else if (total < -tBaseOver2) {
			signal[i] = total + tBase;
			carryOut = -1;
		}
		else {
			signal[i] = total;
			carryOut = 0;
		}
	}
	if (carryOut != 0)
		signal[0] += carryOut;
}

#endif // #ifndef D_IRRBASEBALANCED
