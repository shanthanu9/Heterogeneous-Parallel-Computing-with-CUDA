#include <iostream>
#include <cstdlib>
using namespace std;

#define BLOCK_SIZE 1024
#define INF 1e9
#define CEIL(a, b) ((a-1)/b +1)

__global__ void edge_parallel_bfs(int *depth, int *F, int *C, int n, int m) {

	int id = threadIdx.x;
	
	for(int i = id; i < n; i+=blockDim.x) {
		depth[i] = INF;
	}

	__shared__ int current_depth;
	__shared__ int done;

	if(id == 0) {
		current_depth = 0;
		done = false;
		depth[0] = 0;
	}

	__syncthreads();

	while(!done) {
		
		if(id == 0)
			done = true;

		__syncthreads();

		for(int i = id; i < 2*m; i += blockDim.x) {

			if(depth[F[i]] == current_depth) {
				done = false;
				int v = F[i];
				int u = C[i];
				if(depth[u] > depth[v] + 1) {
					depth[u] = depth[v]+1;
				}
			}
		}

		if(id == 0) {
			current_depth++;
		}

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

	int *h_R = (int*) malloc((n+1)*sizeof(int));

	for(int i = 0; i <= n; i++) {
		cin>>h_R[i];
	}

	int *h_F = (int*) malloc(2*m*sizeof(int));

	for(int i = 0; i < n; i++) {
		for(int j = h_R[i]; j < h_R[i+1]; j++) {
			h_F[j] = i;
		}
	}

	int *h_C = (int*) malloc(2*m*sizeof(int));

	for(int i = 0; i < 2*m; i++) {
		cin>>h_C[i];
	} 

	int *d_F, *d_C, *d_depth;

	cudaMalloc((void**) &d_F, 2*m*sizeof(int));
	cudaMalloc((void**) &d_C, 2*m*sizeof(int));
	cudaMalloc((void**) &d_depth, n*sizeof(int));

	cudaMemcpy(d_F, h_F, 2*m*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, 2*m*sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	edge_parallel_bfs<<<1, BLOCK_SIZE>>>(d_depth, d_F, d_C, n, m);

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
	const int errcount = 20;

	for(int i = 0; i < n; i++) {
		if(h_depth[i] != h_check_depth[i]) {
			flag = false;
			if(count < errcount) {
				cout<<i<<" : "<<h_depth[i]<<" "<<h_check_depth[i]<<endl; 
			}
			count++;
		}
	}

	if(flag) {
		cout<<"Solution is correct!"<<endl;
	}
	else {
		cout<<"Solution is incorrect!"<<endl;
		cout<<count<<" testcases failed."<<endl;
	}

	return 0;
}