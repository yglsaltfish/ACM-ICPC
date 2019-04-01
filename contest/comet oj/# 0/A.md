## 解方程

---

### 题目描述：

给定自然数n，确定关于x，y，z的不定方程$ \sqrt{x-\sqrt{n}} +\sqrt{y}-\sqrt{z}=0​$的所有自然数解。

给出解的数量和所有解对$xyz$之和对$(10^9+7)$取模。

题目链接: [<https://www.cometoj.com/contest/34/problem/A>]

---

### 解题思路：

经过一系列化简可得：$\sqrt{x-\sqrt{n}} +\sqrt{y}-\sqrt{z}=0​$

由此可以推断 : 如果 n 是完全平方数，则有无穷多解，继续化简可得：

$\sqrt{n}=x-(y+z)+\sqrt{4yz}$

由此可知；$x=y+z,n=4yz$ ，因为$x,y,z都是自然数$ 所以$n\%4=0$ ,之后枚举y，就可以得到$xyz$    。

以下是代码

```c++
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
#define db double
const int maxn=1e5+10;
const int mod=1e9+7;
const db eps=1e-9;

int cas;
int  n;
ll res,cnt;
int z,x,y,tmp;

int main()
{
    scanf("%d",&cas);
    while(cas--)
    {
        scanf("%d",&n);
        tmp=sqrt(n);
        if(tmp*tmp-n==0)
        {
            printf("infty\n");
            continue;
        }
        if(n%4 != 0 )
        {
            puts("0 0");
            continue;
        }
        res=0;cnt=0;
        n/=4;
        for(y=1; y*y<=n; y++)
        {
            if(n%y==0 )
            {
                res = (res+(ll)(n/y+y)*n%mod)%mod;
                cnt++;
            }
        }
        printf("%lld %lld\n",cnt,res);
    }
    return 0;
}


```

需要注意的是：for循环内部如果是long long 的话会超时，这题的数据量有点大。









