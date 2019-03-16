#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
using namespace std;

#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))

typedef pair<int,int>  pii;

const int maxn=2e3+10;
const double inf=0x3f3f3f;
const int mod=998244353;
const double eps=1e-9;
using namespace std;

int num[maxn];

int main()
{
    int m,n,k;
    scanf("%d",&m);
    while (m--)
    {
        scanf("%d%d",&n,&k);
        for(int i=0;i<n;++i)
            scanf("%d",&num[i]);
        while(k--)
            next_permutation(num,num+n);
        for (int i=0;i<n;++i)
            printf("%d ",num[i]);
        puts("");
    }
    return 0;
}
