#include <iostream>
#include <cstdio>
using namespace std;

#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))

typedef pair<int ,int>  pii;

const int maxn=2e3+10;
const double inf=0x3f3f3f;
const int mod=998244353;
const double eps=1e-9;
using namespace std;

//priority_queue< pii,vector<pii>,greater<pii> >pq;

int solve(int m,int n)
{
    if(m==0 || n==1)
        return 1;
    else if(m<n)
        return solve(m,m);
    else
        return solve(m,n-1)+solve(m-n,n);
}

int main()
{
    int t,m,n;
    cin>>t;
    while(t--)
    {
        cin>>m>>n;
        cout<<solve(m,n)<<endl;
    }

    return 0;
}
