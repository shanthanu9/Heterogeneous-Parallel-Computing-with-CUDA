#include <iostream>
#include <cstdlib>
using namespace std;

#define BLOCK_SIZE 1024
#define INF 1e9
#define CEIL(a, b) ((a-1)/b +1)

__global__ void edge_parallel_bfs(int *d, int *F, int *C, int n, int m, int *depth) {

	int id = threadIdx.x;
	
	for(int i = id; i < n; i+=blockDim.x) {
		d[i] = INF;
	}

	__shared__ int current_depth;
	__shared__ int done;

	if(id == 0) {
		current_depth = 0;
		done = false;
		d[0] = 0;
	}

	__syncthreads();

	while(!done) {
		
		if(id == 0)
			done = true;

		__syncthreads();

		for(int i = id; i < 2*m; i += blockDim.x) {

			if(d[F[i]] == current_depth) {
				done = false;
				int v = F[i];
				int u = C[i];
				if(d[u] > d[v] + 1) {
					d[u] = d[v]+1;
				}
			}
		}

		if(id == 0) {
			current_depth++;
		}

		__syncthreads();
	}

	if(id == 0)
		*depth = current_depth;
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

	int *d_F, *d_C, *d_d, *d_depth;

	cudaMalloc((void**) &d_F, 2*m*sizeof(int));
	cudaMalloc((void**) &d_C, 2*m*sizeof(int));
	cudaMalloc((void**) &d_d, n*sizeof(int));
	cudaMalloc((void**) &d_depth, sizeof(int));

	cudaMemcpy(d_F, h_F, 2*m*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, 2*m*sizeof(int), cudaMemcpyHostToDevice);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);

	edge_parallel_bfs<<<1, BLOCK_SIZE>>>(d_d, d_F, d_C, n, m, d_depth);

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