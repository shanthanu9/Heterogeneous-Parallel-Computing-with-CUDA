#include <iostream>
#include <cstdlib>
using namespace std;

#define BLOCK_SIZE 1024
#define CEIL(a, b) ((a-1)/b +1)

__global__ void vertex_parallel_bfs(int *depth, int *R, int *C, int n) {

	int id = threadIdx.x;

	for(int i = id; i < n; i+=blockDim.x) {
		depth[i] = 1e9;
	}

	__shared__ int current_depth;
	__shared__ int done;

	if(id == 0) {
		depth[id] = 0;
		current_depth = 0;
		done = false;
	}

	__syncthreads();

	while(!done) {
		if(id == 0)
			done = true;
		
		__syncthreads();

		for(int v = id; v < n; v+=blockDim.x) {
			
			if(depth[v] == current_depth) {

				done = false;
				for(int j = R[v]; j < R[v+1]; j++) {	
					int u = C[j];
					if(depth[u] > depth[v]+1) {
						depth[u] = depth[v]+1;
					}
				}
			}
		}

		if(id == 0)
			current_depth++;

		__syncthreads();
	}
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

	int *d_R, *d_C, *d_depth;

	cudaMalloc((void**) &d_R, (n+1)*sizeof(int));
	cudaMalloc((void**) &d_C, h_R[n]*sizeof(int));
	cudaMalloc((void**) &d_depth, n*sizeof(int));

	cudaMemcpy(d_R, h_R, (n+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, h_R[n]*sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	vertex_parallel_bfs<<<1, BLOCK_SIZE>>>(d_depth, d_R, d_C, n);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	cout<<"Compute time in GPU: "<<milliseconds<<"ms"<<endl;

	int *h_depth = (int*) malloc(n*sizeof(int));

	cudaMemcpy(h_depth, d_depth, n*sizeof(int), cudaMemcpyDeviceToHost);

	int *h_check_depth = (int*)malloc(n*sizeof(int));

	freopen(argv[2], "r", stdin);

	for(int i = 0; i < n; i++) {
		cin>>h_check_depth[i];
	}

	bool flag = true;
	int count = 0;

	for(int i = 0; i < n; i++) {
		if(h_depth[i] != h_check_depth[i]) {
			flag = false;
			count++;
		}
	}

	if(flag) {
		cout<<"Solution is correct!";
	}
	else {
		cout<<"Solution is incorrect!"<<endl;
		cout<<count<<" testcases failed.";
	}

	return 0;
}