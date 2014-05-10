/**
* gpuLucas.cu
*
* A. Thall & A. Hegedus
* Alma College
* 9/1/2010
*
* Implementing the IBDWT method of Crandall in CUDA.
*   This uses a variable base representation and a weighted tranformation
* to reduce the FFT length and eliminate the modular reduction mod M_p.
*
* gpuLucas uses carry-save arithmetic to eliminate carry-adds beyond a single
*   ripple-carry from each digit to the next, following the radix-restoration
*   (dicing) of the convolution products.
*
* The radix-restoration code in IrrBaseBalanced.cu is an ugly kludge,
*   slightly better with templating (thanks, Alex), but it makes up only 1/6th
*   of the runtime, the rest being the weighted transform and componentwise
*   complex multiplication, so might pretty it up, great, but it won't run
*   much faster overall.
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
* Tested:  GTX480 and Tesla C2050, Cuda versions 3.2, 4.0, 4.1
* Compiled with Visual Studio 2008, x64.
*   Uses 64-bit (long long int) and will probably not work in 32-bit x86.
*
* Files:
*    gpuLucas.cu -- main file, including main() and mersenneTest() functions
*    IrrBaseBalanced.cu -- include file (i.e., header, not separate compilation)
*        with the radix-restoration code llintToBalInt<n>() templated routines.
*
* Dependencies:
*    CUFFT
*    cutil library
*    QD extended-precision library for dd_real, double-double class
*        (Computed weights for IBDWT for non-power-of-two FFTs
*         suffered catastrophic cancellation in double.)
*         QD at http://crd-legacy.lbl.gov/~dhbailey/mpdist/
*
* FOR USE:
*    AT COMPILE TIME:
*       1) Set testPrime and signalSize in main()
*                        
*       2) Set setSliceAndDice(x) function in main() to carry high-order bits
*          from x preceeding convolution product digits.  With convolution wordsize
*          typically (18, 19) bits, two preceding terms are typically needed.
*          For shorter wordsizes, a product may need product bits from up to six
*          lower-order words.  setSliceAndDice() assigns a pointer-to-function
*          to a templated function for the chosen number of terms.
*
*    All of this should be altered to be set automatically at runtime based on
*    input testvalue. Most global compile-time dependencies have in fact been
*    eliminated.
*
* Key routines:
*    main() -- sets up the constants for the GPU
*              calls errorTest(testPrime, signalSize), outputs timing and error data
*              calls mersenneTest(testPrime, signalSize) to do full test
*    errorTest(int numIterations, int testPrime, int signalSize)
*    mersenneTest(int testPrime, int signalSize)
*
* Implementing balanced-integers in the irrational base
*
* In bitsPerWord, we use a bit-vector:
*    0 -- low base word
*    1 -- high base word
* Where the positions 0=current, 1=previous, 2=previousprevious, etc.
*    The h_BASE_HI, h_BASE_LO, h_HI_BITS, h_LO_BITS are global constants
* on the host, and BASE_HI, BASE_LO, HI_BITS, LO_BITS, etc., on the device.
* 
* Since minimum word-sizes are (8, 9) in our ranges, never need carry-out bits
* from more than the six preceding terms for a product term, usually, no more than
* two or three with wds of length (18, 19) typical.
*
* NOTE:  must use extended precision to compute A and Ainv for non-power-of-two FFT runlengths
*    We do this on the host using qd library.
*
*  M42643801 took 208299.3 sec/57.86 hours/2.41 days
*    It did 204.72 Lucas iterations per second = 4.88 msec per iteration
*    It used a DWT runlength of 2359296 = 2^21 + 2^18 = 2^18*3^2.
*    and a word-sizes of (18, 19) bits
*    Maximum error reported was 1.8e-1
*  M43112609 to 211447 sec, 58.7 hours, 2.45 days, runlength 2359296
*    M859433 to    112.6 sec, with wd = (17, 18), also with 2 prior words
*   M1257787 to    198.8 sec, with wd = (19, 20), also with 2 prior words, runlength 65536
*                  247.9 sec, on non-overclocked GTX 480
*
*
* Latest build of CUDA and SDK (3.2.12):
*    M859433 to    112.6 sec, with wd = (17, 18), runlength 49152, with 2 prior words
*   M1257787 to    197.2 sec, with wd = (19, 20), also with 2 prior words, runlength 65536 
*                  244.2 sec, on non-overclocked GTX 480
*   M3021377 to   1231.4 sec, with wd = (18, 19), runlength 163840 (2^17+2^15), with 2 prior words
*                                                                 == 2^15*5
*                                                 How about 2^16*3
* 7/2/2011 -- CUDA 4.0
*   M1257787 to    249.2 sec, with wd = (19, 20), two prior words, runlength 65536 on GTX 480
*            to    196.3 sec on Tesla c2050, o/c to 701/1402/1720 Ghz core/proc/mem clocks
*
* 8/1/2011 -- Alex Hegedus and A Thall:
*    Removed CUDPP dependencies
*    Changed to template-based llintToBalInt() for different numbers of carry-bits
*    Rewrote with separate full-test and test-profiling methods
*   M1257787 to    243.5 sec, with wd = (19, 20), two prior words, runlength 65536 on GTX 480
*
* 2/16/2012 -- A Thall
*    CUDA 4.1
*    Can no longer overclock Tesla c2050 with latest NVIDIA GPU control panel
*       No appreciable difference in runtimes between cards...extra processors on 480
*       balance out better floating point performance of Tesla.  480 slightly faster
*       for shorter FFT lengths, tesla for larger
*    Removed broken maxScan code (had been to replace CUDPP dependencies)
*       Replaced with dev-to-host xfer and computation on CPU.
*    MaxScan only used 50 times; if need an actual error-tolerance check for every iteration,
*       write a kernel to simply check each value and do an atomic-set to a signal-flag.
* 2/19/2012 -- A Thall
*    Renamed as gpuLucas
*    Cleanup and documentation of code for release
*    Stripped timing code from mersenneTest()
*
* To do (xxAT: 2/19/2012):
*    1) Need to select testPrime and signalSize variables at runtime
*    2) Need to automatically set llintToBalInt<n>() to correct template at runtime
*    2) Need SIGNAL_SIZE database for space/time trade-offs on CUDA ffts and rebalancing.
*         Really need to auto-tune, since depends on GPU and memory constraints of cards.
*    3) Timing code is a jumbled up mess.
*    5) For a pipelined and double-checked system, need a lot more automagic routines
*         Also need to be able to save current run after X iterations for rechecking,
*         save-and-restart on a multi-user, massively-multi-GPU system.
* 5/10/2014:
*   Removed cutil dependencies for modern Cuda 5.5+ Cudas.
*   Cuda 5.5 Timings:  On NVIDIA Titan, w/ compute capability 3.5
*    M1257787 to   153.8 sec, with wd = (19, 20)
*    M3021377 to   714.7 sec, with wd = (18, 19)
*/

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <qd/dd_real.h>

// includes, project
#include <cufft.h> 
#include <helper_cuda.h>
#include <helper_timer.h>

// Create ThreadsPerBlock constant
const int T_PER_B = 1024;

// NOTE: testPrimes below 9689 generate runlengths < 1024, which breaks the code if T_PER_B = 1024
const int M_9689 = 9689;
const int M_9941 = 9941;
// M_23207 is not prime
const int M_23207 = 23207;
const int M_23209 = 23209;
const int M_44497 = 44497; 
const int M_86243 = 86243; 
const int M_216091 = 216091;
//M_432199 is not a prime, but roughly twice size of M216091
const int M_432199 = 432199;
const int M_756839 = 756839; // M(32)
const int M_859433 = 859433; // M(33) // FFT runlength 49152
const int M_1257787 = 1257787; // M(34)  // FFT runlength 65536 = 2^16
const int M_3021377 = 3021377; // M(36), // FFT runlength 163840 = 2^17 + 2^15 = 5*2^15 // 1998 (GIMPS)
const int M_6972593 = 6972593; // M(38), 1999 (GIMPS)
const int M_13466917 = 13466917; // M(39), 2001 (GIMPS)
const int M_32582657 = 32582657; // M(44), 2006 (GIMPS)
const int M_42643801 = 42643801; // FFT runlength 2359296 = 2^21 + 2^18 = 2^18*3^2.
const int M_43112609 = 43112609; // FFT runlength 2359296 = 2^21 + 2^18 = 2^18*3^2.
// M_86225219 is not prime
const int M_86225219 = 86225219;

/**
 * Currently need to set testPrime and signalSize in main()
 * These were some example values.
 */
//const int TESTPRIME = M_859433; const int SIGNAL_SIZE = 49152;
//const int TESTPRIME = M_1257787; const int SIGNAL_SIZE = 65536;
//const int TESTPRIME = M_3021377; const int SIGNAL_SIZE = 163840;
//const int TESTPRIME = M_42643801; const int SIGNAL_SIZE = 2359296;
//const int TESTPRIME = M_43112609; const int SIGNAL_SIZE = 2359296;

/**
 * The following were scratch-computations to look for good signal-lengths
 *   Need collated lists of time/signal-length/fft trade-offs.
 *   Have earlier CUFFT timings, but need to automate GPU profiling,
 *     and select appropriate lengths at runtime.
 */

/**
 * Using a base 65536 as a starting point, 2^16, giving W = 16 bits
 * with a traditional FFT length = ceil(log2(2*TESTPRIME/log2(2^16))
 *                               = ceil(log2(TESTPRIME/8))
 *                               = ceil(log2(TESTPRIME)) - 3
 *
 * Generally the case that the ibdwt method reduces this by factor of 2,
 *   since don't need to pad prime out to twice nearest power of two
 */
//const int LOG_RUNLENGTH = (int) ceil(log2(1.0*TESTPRIME)) - 4;
//const int SIGNAL_SIZE = (1 << LOG_RUNLENGTH);
//const int SIGNAL_SIZE = 8388608;
//   --- 2**16*5*7
//const int SIGNAL_SIZE = (1 << 23);
//const int SIGNAL_SIZE = 2359296;  // 2**18 * 3**2     ; time 2.947
//const int SIGNAL_SIZE = 2322432;  // 2**12 * 3**4 * 7 ; time 3.004
//const int SIGNAL_SIZE = 2239488;  // 2**10 * 3**7     ; time 2.961

//const int SIGNAL_SIZE = (1 << 24);// + (1 << 22);// - (1 << 18);// + (1 << 18);// - (1 << 17);// 

//const int SIGNAL_SIZE = 1 << (LOG_RUNLENGTH - 1); //  good for M1257787

//const int SIGNAL_SIZE = (1 << (LOG_RUNLENGTH - 1)) + (1 << (LOG_RUNLENGTH - 3)); //  good for M3021377
//const int SIGNAL_SIZE = (1 << (LOG_RUNLENGTH - 1)) + (1 << (LOG_RUNLENGTH - 4)); // good for M43112609, M42643801

// At runtime, set constant and load to GPU for use in IrrBaseBalanced.cu code
int h_LO_BITS;
int h_HI_BITS;
int h_BASE_LO;
int h_BASE_HI;
int h_LO_MODMASK;
int h_HI_MODMASK;

__constant__ int LO_BITS;
__constant__ int HI_BITS;
__constant__ int BASE_LO;
__constant__ int BASE_HI;
__constant__ int LO_MODMASK;
__constant__ int HI_MODMASK;

// Need this include after T_PER_B so can use as shared memory array-length
//   in IrrBaseBalanced.cu routines to avoid dynamic memory alloc on GPU
//   (xxAT sloppy, but okay for now) (means need to recompile for different T_PER_B but
//         have removed NUMBLOCKS dependency, so can do runs of different lengths
// Also needs LO_BITS, etc., constant declarations for templated routines
// This includes all code for parallel carry-add of the balanced-variable base integers
#include "IrrBaseBalanced.cu"

// NOTE:  The largest block size ensures a minimum number of redundant double2llint() functions
//   are called (six extra per block to round product terms and place them in shared memory for
//   "dicing" into individual, variable length words.
//const int NUM_BLOCKS = SIGNAL_SIZE/T_PER_B;  // assume all divisible by T_PER_B

static __host__ void initConstantSymbols(int testPrime, int signalSize) {

	h_LO_BITS = testPrime/signalSize;
	h_HI_BITS = testPrime/signalSize + 1;
	h_BASE_LO = 1 << h_LO_BITS;
	h_BASE_HI = 1 << h_HI_BITS;
	h_LO_MODMASK = h_BASE_LO - 1;
	h_HI_MODMASK = h_BASE_HI - 1;
	checkCudaErrors(cudaMemcpyToSymbol(LO_BITS, &h_LO_BITS, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(HI_BITS, &h_HI_BITS, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(BASE_LO, &h_BASE_LO, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(BASE_HI, &h_BASE_HI, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(LO_MODMASK, &h_LO_MODMASK, sizeof(int)));
	checkCudaErrors(cudaMemcpyToSymbol(HI_MODMASK, &h_HI_MODMASK, sizeof(int)));
}

// Complex data type
typedef cufftDoubleComplex Complex;
typedef cufftDoubleReal Real;
#define CUFFT_TYPEFORWARD CUFFT_D2Z
#define CUFFT_TYPEINVERSE CUFFT_Z2D
#define CUFFT_EXECFORWARD cufftExecD2Z
#define CUFFT_EXECINVERSE cufftExecZ2D

/**
 * PREDECLARED FUNCTIONS:  these don't really need to be predeclared anymore,
 *   but give an overview of the functions so left it.
 */

static __global__ void ComplexPointwiseSqr(Complex*, int);
static __global__ void loadValue4ToFFTarray(double*, int);
static __global__ void loadIntToDoubleIBDWT(double *dArr, int *iArr, int *iHiArr, double *aArr, int size);

/*
 * In bitsPerWord, we use a bit-vector:
 *    0 -- low base word
 *    1 -- high base word
 * Where the positions 0=current, 1=next, 2=nextnext, etc.
 *    The BASE_HI, BASE_LO, HI_BITS, LO_BITS are global constants.
 */
static __host__ void computeBitsPerWord(int testPrime, int *bitsPerWord, int size);
static __host__ void computeBitsPerWordVectors(unsigned char *bitsPerWord8, int *bitsPerWord, int size);

/**
 * code for convolution error-checking
 */
static __global__ void computeErrorVector(float *errorvals, double *fftOut, int size);
static __global__ void computeMaxBitVector(float *dev_errArr, long long *llint_signal, int len);
static __host__ float findMaxErrorHOST(float *dev_fltArr, float *host_temp, int len);

/**
 * compute A and Ainv in extended precision, cast to doubles
 *   and load them to the host arrays.  We include the FFT 1/N scaling with
 *   host_ainv and pull it out of the pointwiseSqrAndScale code
 */
static __host__ void computeWeightVectors(double *host_A, double *host_Ainv, int testPrime, int size);

/**
 * This completes the invDWT transform by multiplying the signal by a_inv,
 *   and subtracts 2 from signal[0], requiring no carry in current weighted carry-save state
 */
static __global__ void invDWTproductMinus2ERROR(long long int *llintArr, double *signal, double *a_inv, int size);
static __global__ void invDWTproductMinus2(long long int *llintArr, double *signal, double *a_inv, int size);


/**
 * The sliceAndDice() function pointer is used to call the correct templated
 *   kernel function to do the distribution of convolution product-bits to
 *   higher-order digits.
 * How many previous convolution components will carry into a given digit depends both
 *   on the base being used and on the length of the convolution vector.  Moreso
 *   on the base...because we are using balanced integers, the product terms don't
 *   scale linearly with the length of the product, but by CLT tend toward a zero
 *   mean with a Gaussian distribution as n gets big.  Average case, but still get
 *   outliers and worst cases. 
 * Use llintToIrrBal<2,3,4,5,6>, as appropriate.  And yes, we can have pointers
 *   to global kernels.  (Works fine, just address.)
 */
void (*sliceAndDice)(int *iArr, int *hiArr, long long int *lliArr, unsigned char *bperW8arr, const int size);

/**
 * For n = 2 to 6. This uses templated kernel functions for the different lengths,
 *   as defined in IrrBaseBalanced.cu file.  (Thanks, Alex.)
 */
void setSliceAndDice(int carryDigits) {

	switch (carryDigits) {
		case 2: sliceAndDice = llintToIrrBal<2>;
		break;
		case 3: sliceAndDice = llintToIrrBal<3>;
		break;
		case 4: sliceAndDice = llintToIrrBal<4>;
		break;
		case 5: sliceAndDice = llintToIrrBal<5>;
		break;
		default: sliceAndDice = llintToIrrBal<6>;
		break;
	}
}

////////////////////////////////////////////////////////////////////////////////
// declaration, forward

/**
 * errorTrial() outputs timing and error information and returns
 *    the average time per Lucas-Lehmer iteration based on timing
 *    of convolution-multiply and rebalancing functions
 */
float errorTrial(int testIterations, int testPrime, int signalSize);
void mersenneTest(int testPrime, int signalSize);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char** argv) 
{
	int deviceCount = 0;
	cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	if (error_id != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id) );
	}
	// This function call returns 0 if there are no CUDA capable devices.
	if (deviceCount == 0)
		fprintf(stderr, "There is no device supporting CUDA\n");
	else
		fprintf(stderr, "Found %d CUDA Capable device(s)\n", deviceCount);

	int dev;
	for (dev = 0; dev < deviceCount; ++dev) {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		fprintf(stderr, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
	}
	fprintf(stderr, "\n and deviceID of max GFLOPS device is %d\n", gpuGetMaxGflopsDeviceId());
	fprintf(stderr, "but we're going to use device 0 by default.\n");
	cudaSetDevice(0);//gpuGetMaxGflopsDeviceId());

	/**
	 * CURRENTLY, SET THESE AT COMPILE TIME
	 */
//	int testPrime = M_1257787;
//	int signalSize = 65536;
	int testPrime = M_3021377;
	int signalSize = 163840;

	// BEGIN by initializing constant memory on device
	initConstantSymbols(testPrime, signalSize);

	// Based on the problem size, and runlength, set the number of carry digits
	//   and assign the global slice-and-dice function from the templated
	//   llintToBalInt<n>() function
	setSliceAndDice(2); // 2 is for the wd (18, 19) used by some typical examples
	/**
	 * END OF COMPILE-TIME SECTION
	 */

	printf("size of long long int = %d (if not 8, you're in trouble)\n", sizeof(long long int));

	printf("Testing M%d, using an irrational base with wordlengths (%d, %d),\n"
		   "giving an FFT runlength of 2^%f = %d\n",
		   testPrime, h_LO_BITS, h_HI_BITS, log(1.0*signalSize)/log(2.0), signalSize);
	printf("  NUM_BLOCKS = %d, T_PER_B = %d\n", signalSize/T_PER_B, T_PER_B);

	StopWatchInterface *timer = NULL;
	sdkCreateTimer(&timer);

	timer->reset();
	timer->start();
	
	// errorTrial() called to give an estimate of convolution sizes and errors,
	//  as well as FFT timings and rebalancing time.
	// return value is average time per Lucas-Lehmer iteration based on
	//   GPU timings
	int trialFraction = 10000;
	float elapsedMsecDEV = errorTrial(testPrime/trialFraction, testPrime, signalSize);

	//get the the total elapsed time in ms

      	timer->stop();
    	float elapsedMsec = timer->getTime();

	printf("\nTiming:  To test M%d"
		   "\n  elapsed time :      %10.f msec = %.1f sec"
		   "\n  dev. elapsed time:  %10.f msec = %.1f sec"
		   "\n  est. total time:    %10.f msec = %.1f sec\n",
		   testPrime,
		   elapsedMsec, elapsedMsec/1000,
		   elapsedMsecDEV*trialFraction, elapsedMsecDEV*trialFraction/1000,
		   elapsedMsecDEV*testPrime, elapsedMsecDEV*testPrime/1000);

	timer->reset();

	printf("\nBeginning full test of M%d\n", testPrime);
	
	timer->start();

	mersenneTest(testPrime, signalSize);

	//get the the total elapsed time in ms
	timer->stop();
    	elapsedMsec = timer->getTime();


	printf("\nTimings:  To test M%d"
		   "\n  elapsed time :      %10.f msec = %.1f sec\n",
		   testPrime, elapsedMsec, elapsedMsec/1000);

	sdkDeleteTimer(&timer);

	cudaThreadExit();	
	exit(0);
}

/**
 * HERE BEGINS THE HOST AND KERNEL CODE TO SUPPORT THE APPLICATION
 *   NOTE:  some changed, moved to IrrBaseBalanced11.cu
 */

// Complex pointwise multiplication
static __global__ void ComplexPointwiseSqr(Complex* cval, int size)
{
	Complex c, temp;
	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < size) {
		temp = cval[tid];
		c.y = 2.0*temp.x*temp.y;
		//c.x = (temp.x + temp.y)*(temp.x - temp.y);  xxAT ??
		c.x = temp.x*temp.x - temp.y*temp.y;
		cval[tid] = c;
	}
} 

/**
 * compute A and Ainv in extended precision, cast to doubles
 *   and load them to the host arrays
 * Uses dd_real 128-bit double-doubles to avoid catastropic cancellation errors
 *   for non-power-of-two FFT lengths
 */
static __host__ void computeWeightVectors(double *host_A, double *host_Ainv, int testPrime, int size) {

	dd_real dd_A, dd_Ainv;
	dd_real dd_N = dd_real(size);
	dd_real dd_2 = dd_real(2.0);

	for (int ddex = 0; ddex < size; ddex++) {
		dd_real dd_expo = dd_real(ddex)*dd_real(testPrime)/dd_N;
		dd_A = pow(dd_2, ceil(dd_expo) - dd_expo);
		dd_Ainv = 1.0/dd_A/dd_N;
		host_A[ddex] = to_double(dd_A);
		host_Ainv[ddex] = to_double(dd_Ainv);
	}
}

static __host__ void computeBitsPerWord(int testPrime, int *bitsPerWord, int size) {

	double PoverN = testPrime/(double)size;
	for (int j = 1; j <= size; j++) {	
		bitsPerWord[j - 1] = (int) (ceil(PoverN*j) - ceil(PoverN*(j - 1)));
	}
}

/**
 * do modular wrap-around to get successive words from element [size - 1]
 * Works backwards to get preceeding bits
 */
static __host__ void computeBitsPerWordVectors(unsigned char *bitsPerWord8, int *bitsPerWord, int size) {
	
	for (int i = 0; i < size; i++) {
		bitsPerWord8[i] = 0;

		for (int bit = 0; bit < 8; bit++) {
			short bitval;
			if (i - bit < 0)
				bitval = (bitsPerWord[size + i - bit] == h_LO_BITS ? 0 : 1);
			else
				bitval = (bitsPerWord[       i - bit] == h_LO_BITS ? 0 : 1);
			bitsPerWord8[i] |= bitval << bit;
		}
	}	
}

// load values of int array into double array for FFT.  Low-order 2 bytes go in lowest numbered
//     position in dArr
static __global__ void loadValue4ToFFTarray(double *dArr, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	if (tid == 0)
		dArr[tid] = 4.0;
	else
		dArr[tid] = 0.0;
}


// This includes pseudobalance by adding hi order terms from last rebalancing.
static __global__ void loadIntToDoubleIBDWT(double *dArr, int *iArr, int *iHiArr, double *aArr, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;
	
	int ival = iArr[tid];
	ival += (tid == 0 ? iHiArr[size - 1] : iHiArr[tid - 1]);

	dArr[tid] = ival*aArr[tid];
}

/**
 * We assume the a_inv also includes the 1/SIGNAL_SIZE scaling needed by the DFT
 * We also do the subtract 2 from the Lucas-square, requiring no carry in the
 *   current balanced carry-save signal.
 */
// Error version assigns non-rounded double value back to signal[tid]
static __global__ void invDWTproductMinus2ERROR(long long int *llintArr, double *signal, double *a_inv, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	double sig;
	if (tid == 0)
		sig = signal[tid]*a_inv[tid] - 2.0;
	else
		sig = signal[tid]*a_inv[tid];

	llintArr[tid] = double2ll(sig, cudaRoundNearest);
	signal[tid] = sig;
}

// Non error version doesn't assign non-rounded double value back to signal[tid]
static __global__ void invDWTproductMinus2(long long int *llintArr, double *signal, double *a_inv, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	double sig;

	if (tid == 0)
		sig = signal[tid]*a_inv[tid] - 2.0;
	else
		sig = signal[tid]*a_inv[tid];
	llintArr[tid] = double2ll(sig, cudaRoundNearest);
}


static __global__ void computeErrorVector(float *errorvals, double *fftOut, int size) {

	const int tid = blockIdx.x*blockDim.x + threadIdx.x;

	double x = fftOut[tid];
	errorvals[tid] = (float) fabs(x - llrint(x));
}

/**
 * uses Xfer to host and then sequential max check on array from errorVector computed above
 *   called seldom (currently, every 1/50 of the total iterations), so no effect on runtime.
 */
static __host__ float findMaxErrorHOST(float *dev_fltArr, float *host_temp, int len) {

	cudaMemcpy(host_temp, dev_fltArr, sizeof(float)*len, cudaMemcpyDeviceToHost);
	float maxVal = 0.0f;
	for (int i = 0; i < len; i++)
		if (host_temp[i] > maxVal)
			maxVal = host_temp[i];
	return maxVal;
}

/**
 *computeMaxVector()
 *function returns list of number of significant bits of a list of long longs
 *AS IS, list can only be as long as however many strings you can launch, now 67,107,840 on 2.0 gpus
 */
static __global__ void computeMaxBitVector(float *dev_errArr, long long *llint_signal, int len){
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	if (tid < len){
		if (llint_signal[tid] >= 0){
			dev_errArr[tid] = (float) __clzll(llint_signal[tid]);
		}
		else{
			dev_errArr[tid] = (float) __clzll(llint_signal[tid]*-1);
		}
	}
}

/**
* errorTrial()
*/
float errorTrial(int testIterations, int testPrime, int signalSize) {

	// We assume throughout that signalSize is divisible by T_PER_B
	const int numBlocks = signalSize/T_PER_B;

	// Allocate host memory to return signal as necessary
	int *h_signalOUT = (int *) malloc(sizeof(int)*signalSize);
 
	// Store computed bit values and bases for precomputation of 
	//    masks for the 
	int *h_bases = (int *) malloc(sizeof(int)*signalSize);
	int *h_bitsPerWord = (int *) malloc(sizeof(int)*signalSize);
	unsigned char *h_bitsPerWord8 = (unsigned char *) malloc(sizeof(unsigned char)*signalSize);

	// Allocate device memory for signal
	int *i_signalOUT;
	Real *d_signal;
	Complex *z_signal;
	int i_sizeOUT = sizeof(int)*signalSize;
	int d_size = sizeof(Real)*signalSize;
	int z_size = sizeof(Complex)*(signalSize/2 + 1);
	int bpw_size = sizeof(unsigned char)*signalSize;

	int llintSignalSize = sizeof(long long int)*signalSize;

	Real *dev_A, *dev_Ainv;
	unsigned char *bitsPerWord8;
	long long int *llint_signal;
	checkCudaErrors(cudaMalloc((void**)&i_signalOUT, i_sizeOUT));
	checkCudaErrors(cudaMalloc((void**)&d_signal, d_size));
	checkCudaErrors(cudaMalloc((void**)&z_signal, z_size));

	checkCudaErrors(cudaMalloc((void**)&dev_A, d_size));
	checkCudaErrors(cudaMalloc((void**)&dev_Ainv, d_size));
	checkCudaErrors(cudaMalloc((void**)&bitsPerWord8, bpw_size));
	checkCudaErrors(cudaMalloc((void**)&llint_signal, llintSignalSize));

	// allocate device memory for DWT weights and base values
	// CUFFT plan
	cufftHandle plan1, plan2;
	checkCudaErrors(cufftPlan1d(&plan1, signalSize, CUFFT_TYPEFORWARD, 1));
	checkCudaErrors(cufftPlan1d(&plan2, signalSize, CUFFT_TYPEINVERSE, 1));

	// Variables for the GPK carry-adder
	// Array for high-bit carry out
	int *i_hiBitArr;
	checkCudaErrors(cudaMalloc((void**)&i_hiBitArr, sizeof(int)*signalSize));

	// CUDPP plan for parallel-scan int GPK adds
	
	//make host and device arrays for error computation
	float *dev_errArr;
	cudaMalloc((void**) &dev_errArr, signalSize*sizeof(float));
	float *host_errArr = (float *) malloc(signalSize*sizeof(float));

	// Compute word-sizes to use when dicing products to sum to int array
	computeBitsPerWord(testPrime, h_bitsPerWord, signalSize);
	computeBitsPerWordVectors(h_bitsPerWord8, h_bitsPerWord, signalSize);
	checkCudaErrors(cudaMemcpy(bitsPerWord8, h_bitsPerWord8, bpw_size, cudaMemcpyHostToDevice));

	for (int i = 0; i < 20; i++) {
		printf("word[%d] numbits = %d\n", i, h_bitsPerWord[i]);
		printf("numbits of this and following 7 are: ");
		for (int bit = 1; bit < 256; bit *= 2)
			printf("%d ", bit & h_bitsPerWord8[i] ? h_HI_BITS : h_LO_BITS);
		printf("\n");
	}
	for (int i = signalSize - 8; i < signalSize; i++) {
		printf("word[%d] numbits = %d\n", i, h_bitsPerWord[i]);
		printf("numbits of this and following 7 are: ");
		for (int bit = 1; bit < 256; bit *= 2)
			printf("%d ", bit & h_bitsPerWord8[i] ? h_HI_BITS : h_LO_BITS);
		printf("\n");
	}
	
	double *h_A = (double *) malloc(signalSize*sizeof(double));
	double *h_Ainv = (double *) malloc(signalSize*sizeof(double));

	// compute weights in extended precision, essential for non-power-of-two signal_size
	computeWeightVectors(h_A, h_Ainv, testPrime, signalSize);
	checkCudaErrors(cudaMemcpy(dev_A, h_A, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_Ainv, h_Ainv, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
	printf("weight vector looks like:\n");
	for (int i = 0; i < 20; i++) 
		printf("a[%d] = %f\n", i, h_A[i]);
	for (int i = 0; i < 20; i++) 
		printf("ainv[%d] = %f\n", i, h_Ainv[i]);

	// load the int array to the doubles for FFT
	// This is already balanced, and already multiplied by a_0 = 1 for DWT
	loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal, signalSize);
	getLastCudaError("Kernel execution failed [ loadValue4ToFFTarray ]");

	float totalTime = 0;
	// Loop M-2 times
    for (unsigned int iter = 2; iter < testIterations; iter++) {
		if (iter % (testIterations/50) == 0) {

			cudaEvent_t start, stop;
			checkCudaErrors(cudaEventCreate(&start));
			checkCudaErrors(cudaEventCreate(&stop));
			checkCudaErrors(cudaEventRecord(start, 0));

			// Transform signal
			checkCudaErrors(CUFFT_EXECFORWARD(plan1, (Real *)d_signal, (Complex *)z_signal));
			getLastCudaError("Kernel execution failed [ CUFFT_EXECFORWARD ]");
			// Multiply the coefficients componentwise
			int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

			ComplexPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);

			getLastCudaError("Kernel execution failed [ ComplexPointwiseSqr ]");

			// Transform signal back
			checkCudaErrors(CUFFT_EXECINVERSE(plan2, (Complex *)z_signal, (Real *)d_signal));
			getLastCudaError("Kernel execution failed [ CUFFT_EXECINVERSE ]");

			checkCudaErrors(cudaEventRecord(stop, 0));
			checkCudaErrors(cudaEventSynchronize(stop));
			float elapsedTime;
			checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
			printf("Time for FFT, squaring, INV FFT:  %3.3f ms\n", elapsedTime);
			totalTime += elapsedTime;
			checkCudaErrors(cudaEventDestroy(start));
			checkCudaErrors(cudaEventDestroy(stop));

			// ERROR TESTS
			invDWTproductMinus2ERROR<<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, signalSize);
			computeErrorVector<<<numBlocks, T_PER_B>>>(dev_errArr, d_signal, signalSize);
			float maxerr = findMaxErrorHOST(dev_errArr, host_errArr, signalSize); 
			printf("\n[%d/50]: iteration %d: max abs error = %f", iter/(testPrime/50), iter, maxerr);

			computeMaxBitVector<<<numBlocks, T_PER_B>>>(dev_errArr, llint_signal, signalSize);
			float maxBitVector = findMaxErrorHOST(dev_errArr, host_errArr, signalSize);
			printf("\n[%d/50]: iteration %d: max Bit Vector = %f", iter/(testPrime/50), iter, maxBitVector);
			fflush(stdout);

			// Time rebalancing
			checkCudaErrors(cudaEventCreate(&start));
			checkCudaErrors(cudaEventCreate(&stop));
			checkCudaErrors(cudaEventRecord(start, 0));
			sliceAndDice<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, llint_signal, bitsPerWord8, signalSize);
			checkCudaErrors(cudaEventRecord(stop, 0));
			checkCudaErrors(cudaEventSynchronize(stop));
			checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
			printf("\nTime to rebalance llint:  %3.3f ms\n", elapsedTime);
			totalTime += elapsedTime;
			checkCudaErrors(cudaEventDestroy(start));
			checkCudaErrors(cudaEventDestroy(stop));
		}
		else {
			// Transform signal
			checkCudaErrors(CUFFT_EXECFORWARD(plan1, (Real *)d_signal, (Complex *)z_signal));
			getLastCudaError("Kernel execution failed [ CUFFT_EXECFORWARD ]");
			// Multiply the coefficients componentwise
			int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

			ComplexPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);
			getLastCudaError("Kernel execution failed [ ComplexPointwiseSqr ]");

			// Transform signal back
			checkCudaErrors(CUFFT_EXECINVERSE(plan2, (Complex *)z_signal, (Real *)d_signal));
			getLastCudaError("Kernel execution failed [ CUFFT_EXECINVERSE ]");

			invDWTproductMinus2<<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, signalSize);
						sliceAndDice<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, llint_signal, bitsPerWord8, signalSize);
		}

		loadIntToDoubleIBDWT<<<numBlocks, T_PER_B>>>(d_signal, i_signalOUT, i_hiBitArr, dev_A, signalSize);
	}
	
	// DONE!  Final copy out from GPU, since not done by default as for CPU stages
	// DO GOOD REBALANCE HERE
	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));
	checkCudaErrors(cudaEventRecord(start, 0));

	addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, signalSize);
	rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signalOUT, bitsPerWord8, signalSize);
	checkCudaErrors(cudaMemcpy(h_signalOUT, i_signalOUT, i_sizeOUT, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaEventRecord(stop, 0));
	checkCudaErrors(cudaEventSynchronize(stop));
	float elapsedTime;
	checkCudaErrors(cudaEventElapsedTime(&elapsedTime, start, stop));
	printf("\nTime to rebalance and write-back:  %3.1f ms\n", elapsedTime);
	checkCudaErrors(cudaEventDestroy(start));
	checkCudaErrors(cudaEventDestroy(stop));

	//Destroy CUFFT context
	checkCudaErrors(cufftDestroy(plan1));
	checkCudaErrors(cufftDestroy(plan2));

	// cleanup memory
	free(h_signalOUT);
	free(h_bases);
	free(h_bitsPerWord);
	free(h_bitsPerWord8);
	free(h_A);
	free(h_Ainv);
	free(host_errArr);

	checkCudaErrors(cudaFree(i_signalOUT));
	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(z_signal));

	checkCudaErrors(cudaFree(i_hiBitArr));
	checkCudaErrors(cudaFree(dev_A));
	checkCudaErrors(cudaFree(dev_Ainv));
	checkCudaErrors(cudaFree(bitsPerWord8));
	checkCudaErrors(cudaFree(llint_signal));

	checkCudaErrors(cudaFree(dev_errArr));
	
	return totalTime/50;
}

/**
* print_residue() -- output the Lucas-Lehmer residue for non-prime exponents
*   needed for result submission to GIMPS, or verifying results with other clients
*/
void print_residue(int testPrime, int *h_signalOUT, int signalSize) {
	static unsigned long int *hex = NULL;
	static unsigned long int prior_hex = 0;
	static char bits_fmt[16] = "\0"; /* "%%0%ulx" -> "%08lx" or "%016lx" depending on sizeof(UL) */
	long long int k, j=0, i, word, k1;
	double lo = floor((exp(floor((double)testPrime/signalSize)*log(2.0)))+0.5);
	double hi = lo+lo;
	unsigned long b = testPrime % signalSize; 
	unsigned long c = signalSize - b; 
	int totalbits = 64;
	
	printf("M_%d, ", testPrime);
	
	int sudden_death = 0; 
	long long int NminusOne = signalSize - 1; 

	while (1) {
			k = j;
			if (h_signalOUT[k] < 0.0) {
					k1 = (j + 1) % signalSize;
					--h_signalOUT[k1];
					if (j == 0 || (j != NminusOne && ((((b*j) % signalSize) >= c) || j == 0)))
							h_signalOUT[k] += hi;
					else
							h_signalOUT[k] += lo;
			} else if (sudden_death)
					break;
			if (++j == signalSize) {
					sudden_death = 1;
					j = 0;
			}
	}

	if (hex != NULL && totalbits/8 + 1 > prior_hex) {
			free(hex);
			hex = NULL;
			prior_hex = totalbits/8 + 1;
	}

	if (hex == NULL && (hex = (unsigned long *)calloc(totalbits/8 + 1, sizeof(unsigned long))) == NULL) {
			printf("Cannot get memory for residue bits; calloc()\n");
			exit(1);
	}
	
	j = 0;
	i = 0;
	do {
			k = (long)(ceil((double)testPrime*(j + 1)/signalSize) - ceil((double)testPrime*j/signalSize));
			if (k > totalbits)
					k = totalbits;
			totalbits -= k;
			word = (long)h_signalOUT[j + ((j & 0) >> 0)];
			for (j++; k > 0; k--, i++) {
					if (i % 8 == 0)
							hex[i/8] = 0L;
					hex[i/8] |= ((word & 0x1) << (i % 8));
					word >>= 1;
			}
	} while(totalbits > 0);
	
	printf("0x");
//	if (bits_fmt[0] != '%')
//			sprintf(bits_fmt, "%%0%lu%s", (unsigned long)(8/4), "lx"); /* 4 bits per hex 'digit' */
	
	for (j = (i - 1)/8; j >= 0; j--) {
			printf("%02lx", hex[j]);
	}
	
	printf(", n = %d, gpuLucas\n", signalSize);
	return;
}

/**
* mersenneTest() -- full test of 2^testPrime - 1, including max error term every 1/50th
*   time through loop
*/
void mersenneTest(int testPrime, int signalSize) {

	// We assume throughout that signalSize is divisible by T_PER_B
	const int numBlocks = signalSize/T_PER_B;
	const int numFFTblocks = (signalSize/2 + 1)/T_PER_B + 1;

	// Allocate host memory to return signal as necessary
	int *h_signalOUT = (int *) malloc(sizeof(int)*signalSize);
 
	// Store computed bit values and bases for precomputation of 
	//    masks for the 
	int *h_bases = (int *) malloc(sizeof(int)*signalSize);
	int *h_bitsPerWord = (int *) malloc(sizeof(int)*signalSize);
	unsigned char *h_bitsPerWord8 = (unsigned char *) malloc(sizeof(unsigned char)*signalSize);

	// Allocate device memory for signal
	int *i_signalOUT;
	Real *d_signal;
	Complex *z_signal;
	int i_sizeOUT = sizeof(int)*signalSize;
	int d_size = sizeof(Real)*signalSize;
	int z_size = sizeof(Complex)*(signalSize/2 + 1);
	int bpw_size = sizeof(unsigned char)*signalSize;

	int llintSignalSize = sizeof(long long int)*signalSize;

	Real *dev_A, *dev_Ainv;
	unsigned char *bitsPerWord8;
	long long int *llint_signal;
	checkCudaErrors(cudaMalloc((void**)&i_signalOUT, i_sizeOUT));
	checkCudaErrors(cudaMalloc((void**)&d_signal, d_size));
	checkCudaErrors(cudaMalloc((void**)&z_signal, z_size));

	checkCudaErrors(cudaMalloc((void**)&dev_A, d_size));
	checkCudaErrors(cudaMalloc((void**)&dev_Ainv, d_size));
	checkCudaErrors(cudaMalloc((void**)&bitsPerWord8, bpw_size));
	checkCudaErrors(cudaMalloc((void**)&llint_signal, llintSignalSize));

	// allocate device memory for DWT weights and base values
	// CUFFT plan
	cufftHandle plan1, plan2;
	checkCudaErrors(cufftPlan1d(&plan1, signalSize, CUFFT_TYPEFORWARD, 1));
	checkCudaErrors(cufftPlan1d(&plan2, signalSize, CUFFT_TYPEINVERSE, 1));

	// Array for high-bit carry out
	int *i_hiBitArr;
	checkCudaErrors(cudaMalloc((void**)&i_hiBitArr, sizeof(int)*signalSize));

	// Error-checking device and host arrays
	float *dev_errArr; 
	cudaMalloc((void**) &dev_errArr, signalSize*sizeof(float));
	float *host_errArr = (float *) malloc(signalSize*sizeof(float));

	// Compute word-sizes to use when dicing products to sum to int array
	computeBitsPerWord(testPrime, h_bitsPerWord, signalSize);
	computeBitsPerWordVectors(h_bitsPerWord8, h_bitsPerWord, signalSize);
	checkCudaErrors(cudaMemcpy(bitsPerWord8, h_bitsPerWord8, bpw_size, cudaMemcpyHostToDevice));

	// compute weights in extended precision, essential for non-power-of-two signal_size,
	//   and then load to device
	double *h_A = (double *) malloc(signalSize*sizeof(double));
	double *h_Ainv = (double *) malloc(signalSize*sizeof(double));
	computeWeightVectors(h_A, h_Ainv, testPrime, signalSize);
	checkCudaErrors(cudaMemcpy(dev_A, h_A, sizeof(double)*signalSize, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(dev_Ainv, h_Ainv, sizeof(double)*signalSize, cudaMemcpyHostToDevice));

	// load the int array to the doubles for FFT
	// This is already balanced, and already multiplied by a_0 = 1 for DWT
	loadValue4ToFFTarray<<<numBlocks, T_PER_B>>>(d_signal, signalSize);
	getLastCudaError("Kernel execution failed [ loadValue4ToFFTarray ]");

	// Loop M-2 times
	for (unsigned int iter = 2; iter < testPrime; iter++) {

		// Transform signal
		checkCudaErrors(CUFFT_EXECFORWARD(plan1, (Real *)d_signal, (Complex *)z_signal));
		getLastCudaError("Kernel execution failed [ CUFFT_EXECFORWARD ]");

		// Multiply the coefficients componentwise
		ComplexPointwiseSqr<<<numFFTblocks, T_PER_B>>>(z_signal, signalSize/2 + 1);
		getLastCudaError("Kernel execution failed [ ComplexPointwiseSqr ]");

		// Transform signal back
		checkCudaErrors(CUFFT_EXECINVERSE(plan2, (Complex *)z_signal, (Real *)d_signal));
		getLastCudaError("Kernel execution failed [ CUFFT_EXECINVERSE ]");

		//    Every 1/50th of the way done, do some error testing
		if (iter % (testPrime/50) == 0) {
			invDWTproductMinus2ERROR<<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, signalSize);
			computeErrorVector<<<numBlocks, T_PER_B>>>(dev_errArr, d_signal, signalSize);
			float maxerr = findMaxErrorHOST(dev_errArr, host_errArr, signalSize);
			printf("\n[%d/50]: iteration %d: max abs error = %f", iter/(testPrime/50), iter, maxerr);
			fflush(stdout);
		}
		else
			invDWTproductMinus2<<<numBlocks, T_PER_B>>>(llint_signal, d_signal, dev_Ainv, signalSize);

		// REBALANCE llint TIMING
		sliceAndDice<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, llint_signal, bitsPerWord8, signalSize);

		loadIntToDoubleIBDWT<<<numBlocks, T_PER_B>>>(d_signal, i_signalOUT, i_hiBitArr, dev_A, signalSize);
	}

	// DONE!  Final copy out from GPU, since not done by default as for CPU stages
	// DO GOOD REBALANCE HERE
	addPseudoBalanced<<<numBlocks, T_PER_B>>>(i_signalOUT, i_hiBitArr, signalSize);
	rebalanceIrrIntSEQGPU<<<1, 1>>>(i_signalOUT, bitsPerWord8, signalSize);
	checkCudaErrors(cudaMemcpy(h_signalOUT, i_signalOUT, i_sizeOUT, cudaMemcpyDeviceToHost));

	bool nonZeros = false;
	for (int i = 0; i < signalSize; i++) {
		if (h_signalOUT[i] > 0) {
			nonZeros = true;
			break;
		}
	}
	if (nonZeros) {
		printf("\nM_%d tests as non-prime.\n", testPrime);

		if (testPrime < 50000) {
			for (int i = 0; i < signalSize; i++) {
			//	unsigned char ch = h_signal[i] & 0xFF;

				printf("%05x", h_signalOUT[i]);
				if (i % 2 == 3)
					printf(" ");
				if (i % 20 == 39)
					printf("\n");
			}
			printf("\n");
		}
		printf("\nM_%d tests as non-prime.\n", testPrime);
		print_residue(testPrime, h_signalOUT, signalSize);
	}
	else
		printf("\nM_%d tests as prime.\n", testPrime);


	//Destroy CUFFT context
	checkCudaErrors(cufftDestroy(plan1));
	checkCudaErrors(cufftDestroy(plan2));

	// cleanup memory
	free(h_signalOUT);
	free(h_bases);
	free(h_bitsPerWord);
	free(h_bitsPerWord8);
	free(h_A);
	free(h_Ainv);
	free(host_errArr);

	checkCudaErrors(cudaFree(i_signalOUT));
	checkCudaErrors(cudaFree(d_signal));
	checkCudaErrors(cudaFree(z_signal));

	checkCudaErrors(cudaFree(i_hiBitArr));
	checkCudaErrors(cudaFree(dev_A));
	checkCudaErrors(cudaFree(dev_Ainv));
	checkCudaErrors(cudaFree(bitsPerWord8));
	checkCudaErrors(cudaFree(llint_signal));

	checkCudaErrors(cudaFree(dev_errArr));
}
