#include <iostream>
#include <cstdlib>
using namespace std;

#define BLOCK_SIZE 1024
#define CEIL(a, b) ((a-1)/b +1)

__global__ void work_efficient_parallel_bfs(int *d, int *R, int *C, int n, int *Q0, int *Q1, int *depth) {

	int id = threadIdx.x;

	for(int i = id; i < n; i+=blockDim.x) {
		d[i] = 1e9;
	}

	__shared__ int Q0_len, Q1_len, current_depth;

	if(id == 0) {
		d[id] = 0;
		Q0[id] = 0;
		Q0_len = 1;
		Q1_len = 0;
		current_depth = 0;
	}	
	
	__syncthreads();

	while(Q0_len) {
		for(int i = id; i < Q0_len; i+=blockDim.x) {
			int v = Q0[i];
			for(int j = R[v]; j < R[v+1]; j++) {
				int u = C[j];
				if(atomicCAS(&d[u], int(1e9), d[v]+1) == int(1e9)) {
					int t = atomicAdd(&Q1_len, 1);
					Q1[t] = u;
				}
			}
		}

		__syncthreads();

		for(int i = id; i < Q1_len; i+=blockDim.x) {
			Q0[i] = Q1[i];
		}

		__syncthreads();

		if(id == 0) {
			Q0_len = Q1_len;
			Q1_len = 0;
			current_depth++;
		}

		__syncthreads();
 	}

 	if(id == 0)
 		*depth = current_depth -1;
}


int main(int argc, char *argv[]) {
	if(argc < 3) {
		cout<<"Expecting a file as command line arguement...";
		return 0;
	}
	
	freopen(argv[1], "r", stdin);

	int n,m;
	cin>>n>>m;

	int *h_R = (int*)malloc((n+1)*sizeof(int));

	for(int i = 0; i <= n; i++) {
		cin>>h_R[i];
	}

	int *h_C = (int*)malloc(h_R[n]*sizeof(int));

	for(int i = 0; i < h_R[n]; i++) {
		cin>>h_C[i];
	} 

	int *d_R, *d_C, *d_d, *Q0, *Q1, *d_depth;

	cudaMalloc((void**) &d_R, (n+1)*sizeof(int));
	cudaMalloc((void**) &d_C, h_R[n]*sizeof(int));
	cudaMalloc((void**) &d_d, n*sizeof(int));
	cudaMalloc((void**) &Q0, n*sizeof(int));
	cudaMalloc((void**) &Q1, n*sizeof(int));
	cudaMalloc((void**) &d_depth, sizeof(int));

	cudaMemcpy(d_R, h_R, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, h_R[n]*sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	work_efficient_parallel_bfs<<<1, BLOCK_SIZE>>>(d_d, d_R, d_C, n, Q0, Q1, d_depth);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout<<"Compute time in GPU: "<<milliseconds<<"ms"<<endl;

	int *h_d = (int*) malloc(n*sizeof(int));
	int *h_depth = (int*) malloc(sizeof(int));

	cudaMemcpy(h_d, d_d, n*sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(h_depth, d_depth, sizeof(int), cudaMemcpyDeviceToHost);

	int *h_check_d = (int*)malloc(n*sizeof(int));

	freopen(argv[2], "r", stdin);

	for(int i = 0; i < n; i++) {
		cin>>h_check_d[i];
	}

	bool flag = true;
	int count = 0;
	const int errcount = 20;

	for(int i = 0; i < n; i++) {
		if(h_d[i] != h_check_d[i]) {
			flag = false;
			if(count < errcount) {
				cout<<i<<" : "<<h_d[i]<<" "<<h_check_d[i]<<endl; 
			}
			count++;
		}
	}

	if(flag) {
		cout<<"Solution is correct!"<<endl;
		cout<<"The depth of the given graph from node 0 is "<<(*h_depth)<<endl;
	}
	else {
		cout<<"Solution is incorrect!"<<endl;
		cout<<count<<" testcases failed."<<endl;
	}

	return 0;
}