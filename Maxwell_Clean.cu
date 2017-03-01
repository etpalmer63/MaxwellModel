
/*
 * Erik Palmer
 * March 1, 2017
 *
 * This is stripped down version for public sharing. This code 
 * simulates an elastic dumbbells based on the 
 * Upper Convective Maxwell (UCM) model. 
 *
 * Global variables are used for important parameters, and 
 * computations are transferred between the CPU (host) and
 * GPU (device) as needed for optimal efficiency.
 *
 * Species switching dynamics have been removed so this 
 * will only simulate the evolution of "active" dumbbells.
 * 
 * To Compile:
 *      nvcc CUDA_FILENAME -lcurand -o EXECUTABLE_NAME 
 * 
 */




#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <errno.h>
#include <ctype.h>

#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
//#include <math_functions.h>


//Define Macros for Error handling

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
	printf("Error at %s:%d\n", __FILE__,__LINE__); \
	return EXIT_FAILURE; }} while(0)
#define CURAND_CALL(x) do { if((x)!= CURAND_STATUS_SUCCESS) { \
	printf("Error at %s:%d\n", __FILE__, __LINE__); \
	return EXIT_FAILURE; }} while(0)

//Define Macro for Histogram debugging
#define PRINT_VAR(x) printf("" #x  "\n ")

//Debugging Macros
#define PRINT_VAR_FLOAT_VALUE(x) printf("" #x "=%f\n", x)
#define PRINT_VAR_INT_VALUE(x) printf("" #x "=%d\n", x) 
//* Also useful: printf("DEBUG LINE %d\n", __LINE__);


//___velocity field on-off matrix ____ 
// note that this matrix is multiplied by the inputted flowrate value
#define U11 0.0
#define U12 0.0 
#define U21 1.0 
#define U22 0.0 
//``````````````````````````````


//___Simulation Constants from paper____
#define INIT_ACT_TO_DNG_RATIO 0.5 
#define LITTLE_D 0.03 				//Default 0.03
#define TAO_FUND 5e-6 				//Default 5e-6
#define ZEE 10.0 	 			//Default 10.0
#define ALPHA_ZERO 0.83				//Default 0.83
#define ALPHA_ONE 0.17				//Default 0.17
#define U_ZERO 14.0      			//Default 14.0
#define D_FREE 12.0				//Default 12.0

//``````````````````````````````



//____Define Global Variables________
//For GPU
__device__ double devStepSizeMicro;
__device__ unsigned int devTimeStepsMicro;
__device__ double devFlowRate;
__device__ double devMaxSpringLength;
__device__ double devFreq;

//For CPU
static long hostNumberOfParticles = 0;
static double hostStepSizeMicro = 0;
static long hostTimeStepsMicro = 0;
static long hostTimeStepsMacro = 0;
static double hostFlowRate;
static double hostMaxSpringLength = 0;
static double hostFreq;
//``````````````````````````````````


/*
 * Function: ParseInput
 * Sorts and examines command line input for inappropriate data
 */
int ParseInput(int argc, char *argv[]){

	int i;

	if (argc != 7){	
		printf("ERROR: Incorrect number of input arguments\n");
		printf("Format: ./Maxwell [number of particles] [micro step size]");
		printf(" [time steps micro] [time steps macro] [flow rate]"); //Max Spring Length Removed
		printf(" [SAOS frequency]\n"); 
		return EXIT_FAILURE; 
	}

	char *argvCopy;



printf("The running program is %s\n", argv[0]);

	for (i=1; i<argc; i++){
		argvCopy = argv[i];

		for (; *argv[i]!='\0'; argv[i]++){
			if (*argv[i]=='.') continue; //skip decimals
			if (isdigit(*argv[i])==0){
				printf("%s is not a number\n", argv[i]);
				return EXIT_FAILURE;
			}
		}
		argv[i] = argvCopy;
	}
	
	errno = 0; 

	hostNumberOfParticles = strtol(argv[1], NULL, 10);
	hostStepSizeMicro = strtod(argv[2], NULL);
	hostTimeStepsMicro = strtol(argv[3], NULL, 10);
	hostTimeStepsMacro = strtol(argv[4], NULL, 10); 	
	hostFlowRate = strtod(argv[5], NULL);
	//hostMaxSpringLength = strtod(argv[6], NULL);  //turned off for UCM Maxwell 
	hostFreq = strtod(argv[6], NULL);
	
	if (hostNumberOfParticles==0){
		printf("Unable to convert %s to positive integer\n", argv[1]);
		return EXIT_FAILURE;
	}
	if (hostTimeStepsMicro==0){
		printf("Unable to convert %s to positive integer\n", argv[3]);
		return EXIT_FAILURE;
	}
	if (hostTimeStepsMacro==0){
		printf("Unable to convert %s to positive integer\n", argv[4]);
		return EXIT_FAILURE;
	}
	if (hostStepSizeMicro==0){
		printf("Unable to convert %s to double\n", argv[2]);
		return EXIT_FAILURE;
	}
	//commented out to allow zero flow rate
	/*
	if (hostFlowRate==0.0){
		printf("Unable to convert %s to positive double\n", argv[5]);
		return EXIT_FAILURE;
	}
	*/

	//commented out for UCM Maxwell
	/*
	if (hostMaxSpringLength == 0){
		printf("Unable to convert %s to positive double\n", argv[6]);
		return EXIT_FAILURE;
	}
	*/
	if (hostFreq == 0){
		printf("Unable to convert %s to positive double\n", argv[6]);
		return EXIT_FAILURE;
	}

	if (errno == ERANGE){
		printf("%s\n", strerror(errno));
		return EXIT_FAILURE;	
	}


	return 0;
}

/*
 * Function PrintSimInfo
 * Prints to terminal information about the current simulation
 */

void PrintSimInfo(){

	// ___ Calculate and output program parameters _____

	printf("___________Running Steady State UCM Maxwell Simulation_________________\n");
	printf("|| Number of Particles: %d\n", hostNumberOfParticles);
	printf("|| Total Time: %g \n", hostTimeStepsMicro * hostStepSizeMicro * hostTimeStepsMacro);
	printf("|| Flow Rate: %g \n", hostFlowRate);
	printf("|| Macro -- Steps: %d, Step Size: %g\n", hostTimeStepsMacro, hostTimeStepsMicro * hostStepSizeMicro);
	printf("|| Micro -- Steps: %d, Step Size: %1.12g\n", hostTimeStepsMicro, hostStepSizeMicro);
	printf("|| Maximum Spring Length: %g\n", hostMaxSpringLength );
	printf("|| SAOS Frequency: %g\n", hostFreq );
	printf(" - - - - - - - - - - - - - - - - - - - - - - - \n");
	
	//``````````````````````````````````````````````````
}


/*
 * Function OutputToFile
 * Writes header containing information about the simulation
 * and contents of three vectors to csv file
 */

void OutputToFile (double XX[], double XY[], double YY[], double time_spent, int count){
	
	
	FILE *OutputFile;
	char OutputFileName[] = "MaxwellSSimData";

	sprintf(OutputFileName, "%s.csv", OutputFileName); //<---Filename

	OutputFile = fopen(OutputFileName, "w");

	if (OutputFile == NULL){
		fprintf(stderr, "Couldn't open output file: %s!\n", OutputFileName);
		exit(1);	
	}

	// ____ Header for textfile _______________________
	

	//Description 	

	fprintf(OutputFile,"**********************************************************************\n");
	fprintf(OutputFile,"*     Simulation For UCM Maxwell                                     *\n"); 
	fprintf(OutputFile,"*                                                                    *\n"); 
	fprintf(OutputFile,"*                                                                    *\n"); 
	fprintf(OutputFile,"*                                                                    *\n"); 
	fprintf(OutputFile,"*                                                                    *\n"); 
	fprintf(OutputFile,"*                                                                    *\n"); 
	fprintf(OutputFile,"**********************************************************************\n");
	
	fprintf(OutputFile,"TotalTime: %3.12g\n", hostTimeStepsMicro * hostStepSizeMicro * hostTimeStepsMacro);
	fprintf(OutputFile,"FlowRate: %g\n", hostFlowRate);
	fprintf(OutputFile,"MacroSteps: %ld\n", hostTimeStepsMacro);
	fprintf(OutputFile,"MacroStepSize: %3.12g\n", hostTimeStepsMicro * hostStepSizeMicro);
	fprintf(OutputFile,"MicroSteps: %ld\n", hostTimeStepsMicro);
	fprintf(OutputFile,"StepSize: %2.12g\n", hostStepSizeMicro);
	fprintf(OutputFile,"NumberOfParticles: %ld\n", hostNumberOfParticles);
	fprintf(OutputFile,"Runtime: %g\n", time_spent);
	fprintf(OutputFile,"MaxSpringLength: %g\n", hostMaxSpringLength);
	fprintf(OutputFile,"SAOSFrequency: %g\n", hostFreq);
	fprintf(OutputFile,"Initial-Active-to-Dangling-Ratio: %g\n", INIT_ACT_TO_DNG_RATIO);
	fprintf(OutputFile,"Potential-well-distance(d): %g\n", LITTLE_D );
	fprintf(OutputFile,"Tao_Fundamental: %g\n", TAO_FUND);
	fprintf(OutputFile,"Z: %g\n", ZEE);
	fprintf(OutputFile,"Alpha_Zero: %g\n", ALPHA_ZERO);
	fprintf(OutputFile,"Alpha_One: %g\n", ALPHA_ONE);
	fprintf(OutputFile,"U_Zero: %g\n", U_ZERO);
	fprintf(OutputFile,"D_Free: %g\n", D_FREE);

	//`````````````````````````````````````````````


	//____ print ensemble average at each macro time step ______ 
	fprintf(OutputFile," - - - - - - - - - - - - - - - - - - - - - - - \n");
	fprintf(OutputFile,"||   XX    ||   XY    ||   YY   ||\n");
	fprintf(OutputFile," - - - - - - - - - - - - - - - - - - - - - - - \n");

	int k;
	for (k=0; k<count; k++){
		fprintf(OutputFile,"% 2.16g,"  , XX[k]);
		fprintf(OutputFile," % 2.16g," , XY[k]);
		fprintf(OutputFile," % 2.16g\n", YY[k]);

	
	}
	//```````````````````````````````````````````````````````

	fclose(OutputFile); 

}


/*
 * Function: 
 * GPU Function
 * Calculates the change of state probability of an active dumbbell 
 * given the spring length
 * Tao must be computed each time: See paper, use equations 10 AND 11.
 */

__device__ double ActiveToDanglingProb (double SpringLen){
        double Tao_zero = TAO_FUND * exp ( U_ZERO ); //Equation (11)  //INEFFICIENT - this computation can be moved out of loop	
	
	//__ HOOK Sim__dimensional__
	double Tao = Tao_zero * exp ( - ( LITTLE_D * LITTLE_D * SpringLen * SpringLen) / U_ZERO ); //Equation (10)
	//````````````````

	return 1.0 - exp( -2.0 * devStepSizeMicro / Tao ); //Equation (13)


}


/*
 * Function: 
 * GPU Function
 * Calculates the change of state probability for a dangling dumbbell. 
 */

__device__ double DanglingToActiveProb (double SpringLen) {

	//__ Hook Sim _____
	return 1.0 - exp( - (ALPHA_ZERO + ALPHA_ONE * SpringLen) * devStepSizeMicro); //Equation (14) 
	//`````````````````

}


/*
 * Function: EvolveActive
 * GPU Function
 * Evolve Active Dumbbell for one micro step on GPU
 */ 

__device__ void EvolveActive (double *SpringLenX, double *SpringLenY, double randx, double randy, double *AvgSpringLifes, double *SimTime,
				double totaltime){
  
	double SpringLenXStep, SpringLenYStep;
	

	double drag_coeff_active = 0.5;  // set to 0.5 for comparison with analytic UCM result
	

	//_____  Non-Dim Evo-Equations  
	
	SpringLenXStep = *SpringLenX //;
		+ (U11 * *SpringLenX + U21 * devFreq * cos(devFreq * *SimTime) * *SpringLenY) * devStepSizeMicro * devFlowRate
		- drag_coeff_active * *SpringLenX * devStepSizeMicro 
		+ sqrt( devStepSizeMicro ) * randx; 
		//````````````````````````````


	SpringLenYStep = *SpringLenY //
		+ (U12 * *SpringLenX + U22 * *SpringLenY) * devStepSizeMicro * devFlowRate
		- drag_coeff_active * *SpringLenY * devStepSizeMicro
		+ sqrt( devStepSizeMicro ) * randy; 
		//```````````````````````````
	
	//``````````````````````````````````````

	*SpringLenX = SpringLenXStep;
	*SpringLenY = SpringLenYStep;

}


/*
 * Function: EvolveDangling
 * GPU Function
 * Evolve Dangling Dumbbell for one micro step on GPU
 */

__device__ void EvolveDangling(double *SpringLenX, double *SpringLenY, double randx, double randy, double *AvgSpringLifes, double *SimTime,
				double totaltime){
  
	double SpringLenXStep, SpringLenYStep;
	
	double drag_coeff_dangle = 0.5; //For comparison with UCM


	//_____ Old Non-Dim Evo-Equations: Hook dumbbells ______
		
	SpringLenXStep = *SpringLenX
		+ (U11 * *SpringLenX + U21 * devFreq * cos(devFreq * *SimTime) * *SpringLenY) * devStepSizeMicro * devFlowRate
		- drag_coeff_dangle * *SpringLenX * devStepSizeMicro
		+ sqrt( drag_coeff_dangle * devStepSizeMicro ) * randx;		
		
	SpringLenYStep = *SpringLenY
		+ (U12 * *SpringLenX + U22 * *SpringLenY) * devStepSizeMicro * devFlowRate
		- drag_coeff_dangle * *SpringLenY * devStepSizeMicro
		+ sqrt( drag_coeff_dangle * devStepSizeMicro ) * randy;
		
	//```````````````````````````````````````````

	*SpringLenX = SpringLenXStep;
	*SpringLenY = SpringLenYStep;

}


/* 
 * Function: Micro_Steps
 * Loops through the Micro loop of the SDE
 */

__global__ void Micro_Steps(	double *SpringLenX, double *SpringLenY, int *SpeciesType,
				curandState *states, curandState *ProbStates,
				double *AvgSpringLifes, double *SimTime, double totaltime){
		
	int i = threadIdx.x + blockIdx.x * blockDim.x;	
	

	//___Device API for Random Number Generation____
	//copy state to local state for efficiency
	curandState localState = states[i];
	curandState localProbState = ProbStates[i];

	int j;	
	

	//TODO: Move node value calculation here, since it only changes once each time this function is called. 

	double2 RandNorm;
    //double RandUniform;
	//double SpringLen;

	for(j=0; j < devTimeStepsMicro; j++){
	
		//generate new random number each time
		RandNorm = curand_normal2_double(&localState);
		//RandUniform = curand_uniform_double(&localProbState);		//Disabled because species switching turned off

		//Calculate Spring Length
		//SpringLen = sqrt(SpringLenX[i] * SpringLenX[i] + SpringLenY[i] * SpringLenY[i]); //Disabled b/c species switching turned off

		

		//_____Evolve Dumbbells According to their species_______
		if (SpeciesType[i]==0){   //if active type
			EvolveActive(&SpringLenX[i], &SpringLenY[i], RandNorm.x, RandNorm.y, AvgSpringLifes, &SimTime[i], totaltime);
		}
		else if (SpeciesType[i]==1){ //if dangling type
			EvolveDangling(&SpringLenX[i], &SpringLenY[i], RandNorm.x, RandNorm.y, AvgSpringLifes, SimTime, totaltime);
		}
		//`````````````````````````````````````````````````````````

		SimTime[i] += devStepSizeMicro; 

	}

	//copy random number generator state back
	states[i] = localState;
	ProbStates[i] = localProbState;

}


/* 
 * Function: RandomGenInit
 * Initialize the random number generator on each of the threads
 * Gives each thread a different seed form *SeedList vector
 */

__global__ void RandomGenInit(unsigned int *SeedList, curandState *states){

	int tid  = blockIdx.x * blockDim.x + threadIdx.x;
	curand_init(SeedList[tid], tid, 0, &states[tid]);
}

__global__ void PrintSpringLengths ( double *SpringLenX, double *SpringLenY) {
	int tid = blockIdx.x * blockDim.x + threadIdx.x;

	printf(" x:% f y:% f \n", SpringLenX[tid], SpringLenY[tid]);
}




/*
 * Function: RndNorm
 * CPU Function to transform uniform random variable [0,1] to normal random variable
 * with mean 0 and Variance defined in the function
 */

__host__ double RndNorm (void)
{
	double Variance = 1; 
	 
	static int HasSpareRandomNum = 0;
	static double SpareRandomNum;

	if(HasSpareRandomNum == 1){
	        HasSpareRandomNum = 0;
		return Variance * SpareRandomNum;  	       
	}

	HasSpareRandomNum = 1;
	
	static double u,v,s;

	do{
		u = (  rand() / ((double) RAND_MAX)) * 2 - 1;
		v = (  rand() / ((double) RAND_MAX)) * 2 - 1;
		s = u * u + v * v;	
	} while (s >= 1 || s == 0); 

	s = sqrt (-2.0 * log(s) / s);
	
        SpareRandomNum = v * s;   //Save spare random number for next function call

	return Variance * u * s;	
}



__host__ void OutputRatio (int Active, int Dangling){
	double ActivePercent =  (double)Active / hostNumberOfParticles; 
	double DanglingPercent = (double)Dangling / hostNumberOfParticles;
	printf("Active %f Dangling: %f\n", ActivePercent, DanglingPercent); 
}



__host__ double AvgSpringLife ( double *SpringLenX, double *SpringLenY, int *SpeciesType){
	int j;
        double Tao_zero = TAO_FUND * exp ( U_ZERO );	
	double Total = 0.0; 
	double SpringLen;
	int ActiveCount = 0;
	for (j=0; j<hostNumberOfParticles; j++){

		if (SpeciesType[j] == 0){	//If active type
		ActiveCount++;
		SpringLen = sqrt( SpringLenX[j] * SpringLenX[j] + SpringLenY[j] * SpringLenY[j]); 
		
		//__Hookean Springs__
		Total += Tao_zero * exp (- LITTLE_D * LITTLE_D * SpringLen * SpringLen / U_ZERO ); 	
		//``````````````````

		}
		
	}
	
	return Total / (double) ActiveCount;

} 





int main(int argc, char *argv[]){
	
	//_____Record Program Run Time
	clock_t begin, end, end2;
	begin = clock();
	double time_spent, time_spent2;
	//````````````````````````````````



	// ____  Read Command Line Arguments _____
		
	if (ParseInput(argc, argv)==EXIT_FAILURE){
		exit(2);
	}
	//`````````````````````````````````````


	PrintSimInfo(); //Output Simulation Variables to Terminal 



	//___ Set Global Variable Values _______
	cudaMemcpyToSymbol(devStepSizeMicro, &hostStepSizeMicro, sizeof(double));
	cudaMemcpyToSymbol(devTimeStepsMicro, &hostTimeStepsMicro, sizeof(unsigned int));
	cudaMemcpyToSymbol(devFlowRate, &hostFlowRate, sizeof(double));
	cudaMemcpyToSymbol(devMaxSpringLength, &hostMaxSpringLength, sizeof(double));
	cudaMemcpyToSymbol(devFreq, &hostFreq, sizeof(double));
	//```````````````````````````````````````


	//____define block and thread structure______
	dim3 block;
	
	if (hostNumberOfParticles < 32){
		block.x = hostNumberOfParticles;
		block.y = 1;	
	}
	else {
		block.x=512;
		block.y = 1;
	}

	dim3 grid ((hostNumberOfParticles + block.x -1) / block.x,1);
	//`````````````````````````````````````



	//__Variables for random number generation on GPU kernels
	curandState *states = NULL;
	curandState *ProbStates = NULL;
	//``````````````````````````````````

	//____allocate memory on GPU for random number generator states______
	CUDA_CALL(cudaMalloc((void **)&states, sizeof(curandState) * hostNumberOfParticles ));
	CUDA_CALL(cudaMalloc((void **)&ProbStates, sizeof(curandState) * hostNumberOfParticles ));
	//`````````````````````````````````````````````````````````````````
	
	//__create vectors of seeds_____
	unsigned int *hostSeeds, *devSeeds;

	unsigned int *hostProbSeeds, *devProbSeeds;


	hostSeeds = (unsigned int *)malloc(hostNumberOfParticles*sizeof(unsigned int));	
	hostProbSeeds = (unsigned int *)malloc(hostNumberOfParticles*sizeof(unsigned int));


	CUDA_CALL(cudaMalloc((void **)&devSeeds, sizeof(unsigned int) * hostNumberOfParticles));
	CUDA_CALL(cudaMalloc((void **)&devProbSeeds, sizeof(unsigned int) * hostNumberOfParticles));
	
	srand(time(NULL));
	
	int i;
	for (i=0; i<hostNumberOfParticles; i++){
		hostSeeds[i] = rand();
		hostProbSeeds[i] = rand();
	}
	//````````````````````````````
	



	CUDA_CALL(cudaMemcpy(devSeeds, hostSeeds, sizeof(unsigned int) * hostNumberOfParticles, cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devProbSeeds, hostProbSeeds, sizeof(unsigned int) * hostNumberOfParticles, cudaMemcpyHostToDevice));
	

	//___initialize kernel random number generator on GPU threads____
	RandomGenInit<<< grid, block >>>(devSeeds, states);
	CUDA_CALL( cudaPeekAtLastError() ); 
	CUDA_CALL( cudaDeviceSynchronize() );
	RandomGenInit<<< grid, block >>>(devProbSeeds, ProbStates);
	CUDA_CALL( cudaPeekAtLastError() ); 
	CUDA_CALL( cudaDeviceSynchronize() );
	//````````````````````````````````````





	//____Spring Length variables____
	double *devSpringLenX, *devSpringLenY; 
	double *hostSpringLenX, *hostSpringLenY;
	//`````````````````````````````````

	//___Dumbbell Species Type Variable___
	int *devSpeciesType; 
	int *hostSpeciesType;
	//``````````````````````````````````

	//_______allocate memory on CPU 
	hostSpringLenX = (double*)malloc(hostNumberOfParticles*sizeof(double));
	hostSpringLenY = (double*)malloc(hostNumberOfParticles*sizeof(double));
	hostSpeciesType = (int*)malloc(hostNumberOfParticles*sizeof(int));
	//`````````````````````````

	//_____allocate memory on GPU for spring length
	CUDA_CALL(cudaMalloc((double**)&devSpringLenX, hostNumberOfParticles*sizeof(double)));
	CUDA_CALL(cudaMalloc((double**)&devSpringLenY , hostNumberOfParticles*sizeof(double))); 
	CUDA_CALL(cudaMalloc((int**)&devSpeciesType, hostNumberOfParticles*sizeof(int)));
	//`````````````````````````````````````


	//___Simulation Time____
	//Variables for tracking time t throughout simulation
	double *devSimTime, *hostSimTime;

	hostSimTime = (double *)malloc(hostNumberOfParticles*sizeof(double));
	CUDA_CALL(cudaMalloc((double**)&devSimTime,hostNumberOfParticles*sizeof(double)));
	//````````````````````````````````````````````



	//___ Set initial Spring Lengths to Normal Distribution
	
	int l; 

	for (l=0; l < hostNumberOfParticles; l++){

		hostSimTime[l] = 0.0;



		//___ Set initial length randomly__
		hostSpringLenX[l] = RndNorm();	//Starting from this appears to speed up
		hostSpringLenY[l] = RndNorm();	// steady state for SAOS	 
		//`````````````````````````````````

		
		//___set initial species type__
		
		hostSpeciesType[l] = 0; //Make all dumbbells active initially

		//`````````````````````````````

	}
	//``````````````````````````````````````````````````


	//____Copy spring lengths to Gpu device
	CUDA_CALL(cudaMemcpy(devSpringLenX, hostSpringLenX, hostNumberOfParticles*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devSpringLenY, hostSpringLenY, hostNumberOfParticles*sizeof(double), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(devSpeciesType, hostSpeciesType, hostNumberOfParticles*sizeof(int), cudaMemcpyHostToDevice));
	//````````````````````````````````````````````````	

	CUDA_CALL(cudaMemcpy(devSimTime, hostSimTime, hostNumberOfParticles*sizeof(double), cudaMemcpyHostToDevice));



	//PrintSpringLengths<<< grid, block >>>(devSpringLenX, devSpringLenY); //print lengths to verify created correctly




	//___ initialize variables to calculate and store ensemble average 
	double *Spring_AvgLen_XX;
	double *Spring_AvgLen_XY;
	double *Spring_AvgLen_YY;

	Spring_AvgLen_XX = (double*)malloc((hostTimeStepsMacro+1)*sizeof(double));
	Spring_AvgLen_XY = (double*)malloc((hostTimeStepsMacro+1)*sizeof(double));
	Spring_AvgLen_YY = (double*)malloc((hostTimeStepsMacro+1)*sizeof(double));

	int k;

	double EnsembleAverageXX_Active = 0.0;
	double EnsembleAverageXY_Active = 0.0;
	double EnsembleAverageYY_Active = 0.0;
	
	double EnsembleAverageXX_Dangling = 0.0;
	double EnsembleAverageXY_Dangling = 0.0;
	double EnsembleAverageYY_Dangling = 0.0;


	int j;
	//````````````````````````````````````````````````````````



	int NumberOfActive = 0;
	int NumberOfDangling = 0;

	//_____calculate ensemble average at time = 0 
	for (j=0; j<hostNumberOfParticles; j++){

		if (hostSpeciesType[j]==0){ //if dumbbell is Active type
			NumberOfActive++;
						
			//___Hookean Springs____
			
			EnsembleAverageXX_Active += - hostSpringLenX[j] * hostSpringLenX[j];
			EnsembleAverageXY_Active += - hostSpringLenX[j] * hostSpringLenY[j];
			EnsembleAverageYY_Active += - hostSpringLenY[j] * hostSpringLenY[j];
			
			//```````````````````
			

		} else if (hostSpeciesType[j]==1){ //if dumbbell is Dangling type
			NumberOfDangling++;
						
			//___Hookean Springs____
			
			EnsembleAverageXX_Dangling += -hostSpringLenX[j] * hostSpringLenX[j];
			EnsembleAverageXY_Dangling += -hostSpringLenX[j] * hostSpringLenY[j];
			EnsembleAverageYY_Dangling += -hostSpringLenY[j] * hostSpringLenY[j];
			
			//```````````````````
			
		} else {
			printf("Error1: Unable to Classify Species Type\n");
		}
	}




	if (NumberOfActive == 0){
		
		Spring_AvgLen_XX[0] = EnsembleAverageXX_Dangling / (double)NumberOfDangling;
		Spring_AvgLen_XY[0] = EnsembleAverageXY_Dangling / (double)NumberOfDangling;
		Spring_AvgLen_YY[0] = EnsembleAverageYY_Dangling / (double)NumberOfDangling;

	} else if ( NumberOfDangling == 0){
	
		Spring_AvgLen_XX[0] = EnsembleAverageXX_Active / (double)NumberOfActive;
		Spring_AvgLen_XY[0] = EnsembleAverageXY_Active / (double)NumberOfActive;
		Spring_AvgLen_YY[0] = EnsembleAverageYY_Active / (double)NumberOfActive;
	
	} else {

	Spring_AvgLen_XX[0] = EnsembleAverageXX_Active / (double)NumberOfActive + EnsembleAverageXX_Dangling / (double)NumberOfDangling;
	Spring_AvgLen_XY[0] = EnsembleAverageXY_Active / (double)NumberOfActive + EnsembleAverageXY_Dangling / (double)NumberOfDangling;
	Spring_AvgLen_YY[0] = EnsembleAverageYY_Active / (double)NumberOfActive + EnsembleAverageYY_Dangling / (double)NumberOfDangling;
	
	}



	//``````````````````````````````````
	
	//____To Calculate Average Length of all Active Dumbbells___
	double *hostAverageSpringLife, *devAverageSpringLife;
	
	hostAverageSpringLife = (double *)malloc(sizeof(double));
	CUDA_CALL(cudaMalloc((double**)&devAverageSpringLife,sizeof(double))); 
	//```````````````````````````````````````````````````````

	double totaltime = hostStepSizeMicro * hostTimeStepsMicro * hostTimeStepsMacro;





	//_____ Main simulation loop (Macro Time) ____


	for (k=1; k<=hostTimeStepsMacro; k++){



		//Calculate Average Length of all Active dumbbells		
		*hostAverageSpringLife = AvgSpringLife(hostSpringLenX, hostSpringLenY, hostSpeciesType); 
		CUDA_CALL(cudaMemcpy(devAverageSpringLife,hostAverageSpringLife,sizeof(double),cudaMemcpyHostToDevice));
		
		//Call function to perform computations on GPU
		Micro_Steps<<<grid,block>>>(devSpringLenX,devSpringLenY,devSpeciesType,states,ProbStates,devAverageSpringLife,devSimTime,totaltime);

		//read result from gpu(device) back to cpu(host)
		CUDA_CALL(cudaMemcpy(hostSpringLenX, devSpringLenX, hostNumberOfParticles*sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(hostSpringLenY, devSpringLenY, hostNumberOfParticles*sizeof(double), cudaMemcpyDeviceToHost));
		CUDA_CALL(cudaMemcpy(hostSpeciesType, devSpeciesType, hostNumberOfParticles*sizeof(int), cudaMemcpyDeviceToHost));
	
		//read sim time back from gpu(device) back to cpu(host)
		CUDA_CALL(cudaMemcpy(hostSimTime, devSimTime, sizeof(double), cudaMemcpyDeviceToHost)); 
		


		NumberOfActive = 0;
		NumberOfDangling = 0;

		EnsembleAverageXX_Active = 0;
		EnsembleAverageXY_Active = 0;
		EnsembleAverageYY_Active = 0;
		EnsembleAverageXX_Dangling = 0;
		EnsembleAverageXY_Dangling = 0;
		EnsembleAverageYY_Dangling = 0;



		//____ Calculate Ensemble Averages ______
		for (j=0; j<hostNumberOfParticles; j++){


			if (hostSpeciesType[j]==0){ //if dumbbell is Active type
				NumberOfActive++;
				
				//___Hookean Springs____
				
				EnsembleAverageXX_Active += -hostSpringLenX[j] * hostSpringLenX[j];
				EnsembleAverageXY_Active += -hostSpringLenX[j] * hostSpringLenY[j];
				EnsembleAverageYY_Active += -hostSpringLenY[j] * hostSpringLenY[j];
				
				//```````````````````
				

			} else if (hostSpeciesType[j]==1){ //if dumbbell is Dangling type
				NumberOfDangling++;
				//____Hookean Springs_____
					
				EnsembleAverageXX_Dangling += -hostSpringLenX[j] * hostSpringLenX[j];
				EnsembleAverageXY_Dangling += -hostSpringLenX[j] * hostSpringLenY[j];
				EnsembleAverageYY_Dangling += -hostSpringLenY[j] * hostSpringLenY[j];
				
				//````````````````````````````	

			} else {
				printf("Error2: Unable to Classify Species Type of Dumbbell[%d] with Type: %d \n", j, hostSpeciesType[j]);
				exit(4);
			}

		}
		

		if (NumberOfActive == 0){
			
			Spring_AvgLen_XX[k] = EnsembleAverageXX_Dangling / (double)NumberOfDangling;
			Spring_AvgLen_XY[k] = EnsembleAverageXY_Dangling / (double)NumberOfDangling;
			Spring_AvgLen_YY[k] = EnsembleAverageYY_Dangling / (double)NumberOfDangling;

		} else if ( NumberOfDangling == 0){
		
			Spring_AvgLen_XX[k] = EnsembleAverageXX_Active / (double)NumberOfActive;
			Spring_AvgLen_XY[k] = EnsembleAverageXY_Active / (double)NumberOfActive;
			Spring_AvgLen_YY[k] = EnsembleAverageYY_Active / (double)NumberOfActive;
		
		} else {
 
		Spring_AvgLen_XX[k] = EnsembleAverageXX_Active / (double)NumberOfActive + EnsembleAverageXX_Dangling / (double)NumberOfDangling;
		Spring_AvgLen_XY[k] = EnsembleAverageXY_Active / (double)NumberOfActive + EnsembleAverageXY_Dangling / (double)NumberOfDangling;
		Spring_AvgLen_YY[k] = EnsembleAverageYY_Active / (double)NumberOfActive + EnsembleAverageYY_Dangling / (double)NumberOfDangling;
		
		}




	}
	//``````````````End Macro loop``````````````



	// __ stop computational clock ____	
	end = clock();
	time_spent = double(end-begin)/ CLOCKS_PER_SEC;
	//````````````````````````````````


	//___Write Values to .csv file
	OutputToFile(Spring_AvgLen_XX, Spring_AvgLen_XY, Spring_AvgLen_YY, time_spent, k);
	//````````````````````````````

	
	OutputRatio(NumberOfActive,NumberOfDangling);

	//___ clean up memory ____

	free(hostSimTime);
	CUDA_CALL(cudaFree(devSimTime)); 

	free(hostAverageSpringLife);
	CUDA_CALL(cudaFree(devAverageSpringLife));

	
	free(hostSeeds);
	CUDA_CALL(cudaFree(devSeeds));
	CUDA_CALL(cudaFree(states));
	
	free(hostProbSeeds);
	CUDA_CALL(cudaFree(devProbSeeds));
	CUDA_CALL(cudaFree(ProbStates));
	
	free(hostSpringLenX);
	free(hostSpringLenY);
	CUDA_CALL(cudaFree(devSpringLenX));
	CUDA_CALL(cudaFree(devSpringLenY));



	free(Spring_AvgLen_XX);
	free(Spring_AvgLen_XY);
	free(Spring_AvgLen_YY);

	
	//```````````````````````

	cudaDeviceReset();


	// __ stop computational clock ____	
	end2 = clock();
	time_spent2 = double(end2-begin)/ CLOCKS_PER_SEC;
	printf("Runtime: %f\n\n", time_spent2);
	//````````````````````````````````
	 

	return EXIT_SUCCESS; 

}


