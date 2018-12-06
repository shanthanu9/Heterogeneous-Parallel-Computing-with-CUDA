// Serial BFS

#include <iostream>
#include <vector>
#include <queue>
#include <chrono>
#include <cstdlib>

using namespace std;

int main(int argc, char *argv[]) {
	if(argc < 2) {
		cout<<"Expecting a file as command line arguement...";
		return 0;
	}
	
	freopen(argv[1], "r", stdin);

	int n,m;
	cin>>n>>m;

	vector<vector<int> > adj(n, vector<int>());

	int prev;
	int *number_of_edges = (int*)malloc(n*sizeof(int));
	cin>>prev;

	for(int i = 0; i < n; i++) {
		int temp;
		cin>>temp;
		number_of_edges[i] = temp-prev;
		prev = temp;
	}

	for(int i = 0; i < n; i++) {
		for(int j = 0; j < number_of_edges[i]; j++) {
			int temp;
			cin>>temp;
			fflush(stdout);
			adj[i].push_back(temp);
		}
	}

	queue<int> q;
	int *depth = (int*)malloc(n*sizeof(int));
	for(int i = 0; i < n; i++) depth[i] = 1e9;
	bool *vis = (bool*)malloc(n*sizeof(bool));
	for(int i = 0; i < n; i++) vis[i] = false;
	q.push(0);
	vis[0] = true;
	depth[0] = 0;

	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

	while(!q.empty()) {
		int v = q.front();
		q.pop();
		for(auto u:adj[v]) {
			if(!vis[u]) {
				vis[u] = true;
				q.push(u);
				depth[u] = depth[v]+1;
			}
		}
	}

	std::chrono::steady_clock::time_point end= std::chrono::steady_clock::now();

	cout<<"Time taken for serial BFS: "<<(std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count())<<"ms\n";

	freopen(argv[2], "w", stdout);

	for(int i = 0; i < n; i++) {
		cout<<depth[i]<<endl;
	}

	return 0;
}