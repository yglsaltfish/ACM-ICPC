### 1、最大公约数 最小公倍数

```c++
ll gcd(ll a,ll b){return b==0?a:gcd(b,a%b);}
ll lcm(lla ,ll b){return min(a,b)*gcd(a,b);}
```

### 2、线性素数筛法

```c++
//cnt为素数个数
long long prime[MAX],cnt;
bool isprime[MAX];
void getprime()
{
    cnt=1;
    memset(isprime,1,sizeof(isprime));
    isprime[0]=isprime[1]=0;
    for(long long i=2; i<=MAX; i++)
    {
        if(isprime[i])
            prime[cnt++]=i;
        for(long long j=0; j<cnt&&prime[j]*i<MAX; j++)
        {
            isprime[prime[j]*i]=0;
        }
    }
}
```

### 3、欧拉函数

```c++
const int maxn = 1e6 + 1000;
bool vis[maxn];
int phi[maxn], prim[maxn/10],cnt;
void gephi(){
    phi[1] = 1;
    for(int i=2;i<=maxn/10;i++){
        if(!vis[i]) {
            phi[i] = i-1;
            prim[++cnt] = i;
        }
        for(int j=1;j<=cnt;j++){
            int tp = prim[j];
            if(i*tp>maxn/10) break;
            vis[i*tp]=true;
            if(i%tp==0) {
                phi[i*tp]=phi[i]*tp;
                break;
            }else phi[i*tp]=phi[i]*phi[tp];
        }
    }
}
```

### 4、矩阵快速幂

```c++
//首先定义一下矩阵类型
typedef struct matrixnod
{
    int m[3][3];
} matrix;
//3*3的矩阵乘法
matrix mat(matrix a,matrix b)
{
    matrix c;
    int mod;
    for (int i=0;i<3;i++)
    for (int j=0;j<3;j++)
    {
        c.m[i][j]=0;
        for (int k=0;k<3;k++) c.m[i][j]+=(a.m[i][k]*b.m[k][j])%mod;
        c.m[i][j]%=mod;
    }
    return c;
}
//矩阵快速幂 b^n
matrix doexpmat(matrix b,int n)
{
    matrix a= //单位矩阵
    {
        1,0,0,
        0,1,0,
        0,0,1
    };
    while(n)
    {
        if (n&1) a=mat(a,b);
        n=n>>1;
        b=mat(b,b);
    }
    return a;
}

```

### 5、线段树

##### (1)单点更新，区间查询

```c++
int segm[maxn<<2];
void pushup(int i)
{
	segm[i]=segm[i<<1]+segm[i<<1 | 1];
}
void build(int root,int l ,int r)
{
	if(l==r)
	{
		scanf("%d",&segm[root]);
		return ;
	}
	int m=l+((r-l)>>1);
	build(root<<1,l,m);
	build(root<<1|1,m+1,r);
	pushup(root);
}
void update(int id ,int val,int i,int l ,int r)
{
	if(l==r)
	{
		segm[i]+=val;
		return ;
	}
	int m=l+((r-l)>>1);
	if(id<=m)
		update(id,val,i<<1,l,m);
	if(id>m)
		update(id,val,i<<1|1,m+1,r);
	pushup(i);
}
int query(int ql,int qr,int i,int l,int r)
{
	if(ql<=l && qr>=r)
		return segm[i];
	int m=l+((r-l)>>1),ans=0;
	if(ql<=m)
		ans+=query(ql,qr,i<<1,l,m);
	if(qr>m)
		ans+=query(ql,qr,i<<1|1,m+1,r);
	return ans;
}	
```

##### (2)区间更新，区间查询

```c++
struct seg{
    int l,r;
    int dat,lazy;
    seg(){lazy=0;}
}segm[maxn<<2];

void build(int p,int l,int r)
{
    segm[p].l=l;
    segm[p].r=r;
    if(l==r)
    {
        segm[p].dat=0;
        return ;
    }

    int m=l+((r-l)>>1);
    build(p<<1,l,m);
    build(p<<1|1,m+1,r);
    segm[p].dat=segm[p<<1|1].dat+segm[p<<1].dat;

    return ;
}

void spread(int p)
{
    if(segm[p].lazy)
    {
        segm[p<<1].dat+=segm[p].lazy*(segm[p<<1].r-segm[p<<1].l+1);
        segm[p<<1|1].dat+=segm[p].lazy*(segm[p<<1|1].r-segm[p<<1|1].l+1);
        segm[p<<1].lazy+=segm[p].lazy;
        segm[p<<1|1].lazy+=segm[p].lazy;
        segm[p].lazy=0;
    }
    return ;
}

void update(int p,int val,int l,int r)
{
    if(l<=segm[p].l && r>=segm[p].r)
    {
        segm[p].lazy+=val;
        segm[p].dat+=val*(segm[p].r-segm[p].l+1);
        return ;
    }
    spread(p);
    int m=segm[p].l+((segm[p].r-segm[p].l>>1));
    if(l<=m) update(p<<1,val,l,r);
    if(r>m) update(p<<1|1,val,l,r);

    segm[p].dat=segm[p<<1|1].dat+segm[p<<1].dat;

    return ;
}

int query(int p,int l,int r)
{
    if(l<=segm[p].l &&segm[p].r<=r)
    {
        return segm[p].dat;
    }
    spread(p);
    int ans=0;
    int mid=segm[p].l+((segm[p].r-segm[p].l>>1));
    if(l<=mid ) ans+=query(p<<1,l,r);
    if(r>mid)  ans+=query(p<<1|1,l,r);
    return ans;
}
```

### 6、并查集

```c++
int father[250010 * 2], rank[250010 * 2];
void disjoint_set(int n)
{
    for(int i = 1; i <= n; i++)
        father[i] = i;
}
int find(int v)
{
    return father[v] = father[v] == v? v: find(father[v]);
}
void merge(int x, int y)
{
    int a = find(x), b = find(y);
    if(rank[a] < rank[b])
        father[a] = b;
    else
    {
        father[b] = a;
        if(rank[a] == rank[b])
            rank[a]++;
    }
}
```

### 7、计算几何

```c++
using namespace std;
#define db double
#define ll long long

const db eps=1e-9;
int sign(db a)
{ 
    if(abs(a)<eps)
        return 0;
    return a < 0 ? -1 : 1;
}
int cmp(db a, db b){return sign(a-b);}

struct Point{
    db x,y;
    Point() {}
    Point(db _x,db _y):x(_x),y(_y){}
};

typedef Point Vec;
// 向量+向量 = 向量 ，点 + 向量 = 点 
Vec operator+(Vec A, Vec B) { return Vec(A.x + B.x, A.y + B.y); }

// 点 - 点 = 向量 
Vec operator-(Point A, Point B) { return Vec(A.x - B.x, A.y - B.y); }

//向量 * 数 = 向量
Vec operator*(Vec A, db p) { return Vec(A.x * p, A.y * p); }

//向量 / 数 = 向量
Vec operator/(Vec A, db p) { return Vec(A.x / p, A.y / p); }

bool operator == (const Point &A ,const Point &B)
{
    return sign(A.x - B.x) == 0 && sign(A.y - B.y) == 0;
}

struct Line{
    Point a,b;
    Line(){}
    Line(Point _a,Point _b):a(_a),b(_b){}
};

/*  两点间距离   */
db get_distance(Point a, Point b){return sqrt((a.x-b.x)*(a.x-b.x)+(a.y-b.y)*(a.y-b.y));}

//  两向量的叉乘
db Cross(Vec A,Vec B){return A.x * B.y - A.y * B.x;}
db Area2(Point A,Point B,Point C){return Cross(B - A, C - A);}

//  两向量的点乘
db Dot(Vec A,Vec B){return A.x * B.x + A.y * B.y;}

//向量长度
db Length(Vec A){return sqrt(Dot(A, A));}

//向量角度
db angle(Vec A,Vec B){return acos(Dot(A, B) / Length(A) / Length(B));}

//求向量的单位法线(A不能是零向量)
Vec Normal(Vec A){db L = Length(A);return Vec(-A.y / L, A.x / L);}

//求向量旋转 , 逆时针旋转角度rad
Vec Rotate(Vec A,db rad)
{
    return Vec(A.x * cos(rad) - A.y * sin(rad), A.x * sin(rad) + A.y * cos(rad));
}

//求点到直线的距离
db dis_dot_Line(Point P,Point A,Point B)
{
    Vec v1 = B - A, v2 = P - A;
    return abs(Cross(v1, v2) / Length(v1));
}

//求点到直线的投影
Point GetLineProjection(Point P,Point A ,Point B)
{
    Vec v = B - A;
    return A + v * (Dot(v, P - A) / Dot(v, v));
}

//线段相交  (规范相交)
bool SegmentIntersection(Point A1,Point A2,Point B1,Point B2)
{
    db c1 = Cross(A2 - A1, B1 - A1), c2 = Cross(A2 - A1, B2 - A1);
    db c3 = Cross(B2 - B1, A1 - B1), c4 = Cross(B2 - B1, A2 - B1);
    return sign(c1) * sign(c2) < 0 && sign(c3) * sign(c4) < 0;
}

//判断点是否在线段上（不包含端点）
bool onSegment(Point p,Point a1,Point a2)
{
    return sign(Cross(a1 - p, a2 - p)) == 0 && sign(Dot(a1 - p, a2 - p));
}

//点到线段的距离
db distancetosegment(Point P,Point A,Point B)
{
    if(A==B)
        return Length(P - A);
    Vec v1 = B - A, v2 = P - A, v3 = P - B;
    if(sign(Dot(v1,v2))<0 )
        return Length(v2);
    else if(sign(Dot(v1,v3))>0)
        return Length(v3);
    else
        return abs(Cross(v1, v2)) / Length(v1);
}

int main()
{
    ios::sync_with_stdio(false);
    
    return 0;
}

```



### 8、RMQ 算法(ST表)

```c++
void ST()
{
    for(int i=1;i<=n;i++) RMQ[i][0]=code[i];
    for(int j=1;(1<<j)<=n;j++)
        for(int i=1;i+(1<<j)-1<=n;i++)
           RMQ[i][j]=max(RMQ[i][j-1],RMQ[i+(1<<(j-1))][j-1]);
}
int Query(int L,int R)
{
    int k=0;
    while((1<<(k+1))<=R-L+1) k++;
    return max(RMQ[L][k],RMQ[R-(1<<k)+1][k]);
}
```

### 9、最小生成树

##### (1)prime算法

```c++
struct edge
{
    int to ;
    ll cost;
    edge (int tt,int cc) :to(tt),cost(cc) {}
    edge() {}
    bool operator <(const edge &a)const
    {
        return a.cost <cost;
    }
};
priority_queue<edge> que;
vector <edge> G[maxn];
bool vis[maxn];

ll prime()
{
    ll res=0;
    for(int i=0; i<G[1].size(); i++)
        que.push(G[1][i]);
    vis[1]=true;
    while(!que.empty())
    {
        edge e=que.top();
        que.pop();
        if(vis[e.to])
            continue;
        vis[e.to]=true;
        res+=e.cost;
        for(int i=0; i<G[e.to].size(); i++)
            que.push(G[e.to][i]);
    }
    return res;
}
```

##### (2) kruscal 算法

```c++
struct edge{
    int from,to;
    ll cost;
    edge(int tt,ll cst) :to(tt),cost(cst){}
    edge(){}
}G[maxn];

bool cmp(edge a,edge b)
{
    return a.cost<b.cost;
}

void init()
{
    for(int i=1;i<=n;i++)
      fa[i]=i;
}

int find_f(int x)
{
    if(x==fa[x]) return x;
    else
        return x=find_f(fa[x]);
}

bool same(int x,int y)
{
    return find_f(x)==find_f(y);
}

void unio(int x,int y)
{
    int u=find_f(x),v=find_f(y);
    if(u==v) return;
    else
        fa[u]=v;
}
ll kruscal()
{
    ll res=0;
    sort(G+1,G+1+n,cmp);
    for(int i=1;i<=n;i++)
    {
        if(!same(G[i].from,G[i].to))
        {
            unio(G[i].from,G[i].to);
            res+=G[i].cost;
        }
    }
    return res;
}
```

### 10、trie树

```c++
struct Trie
{

    int next[maxn][26], ed[maxn];
    int L, root;
    int newnode()
    {
        for(int i = 0; i < 26; i++)
            next[L][i] = -1;
        ed[L] = 0;
        return L++;
    }
    void init()
    {
        L = 0;
        root = newnode();
    }
    void insert(char s[])
    {
        int now = root;
        for(int i = 0, sz = strlen(s); i < sz; i++)
        {
            if(next[now][s[i] - 'A'] == -1)
                next[now][s[i] - 'A'] = newnode();
            now = next[now][s[i] - 'A'];
        }
        ed[now] = 1;
    }
    
    bool query(char s[])
    {
        int now = root;
        for(int i = 0, sz = strlen(s); i < sz; i++)
        {
            if(next[now][s[i] - 'A'] == -1)
            {
                return false;
            }
            now = next[now][s[i] - 'A'];
        }
        return ed[now] == 1;
    }

};
```

### 11、主席树

##### (1)区间第K大

```c++
int n, m;
int tot=0;
int root[maxn];
struct node
{
    int sum;
    int l,r;
} p[maxn*20];

int build(int l,int r)
{
    int rt=++tot;
    p[rt].sum=0;
    p[rt].l=l,p[rt].r=r;
    if(l==r)
        return rt;
    int mid=l+(r-l>>1);
    p[rt].l=build(l,mid);
    p[rt].r=build(mid+1,r);
    return rt;
}

int update(int l,int r,int pre,int k)
{
    int rt=++tot;
    p[rt]=p[pre];
    p[rt].sum++;
    int mid=l+(r-l>>1);
    if(l==r) return rt;
    if(mid>=k) p[rt].l=update(l,mid,p[pre].l,k);
    else p[rt].r=update(mid+1,r,p[pre].r,k);
    return rt;
}

int query(int l,int r,int x,int y,int k)
{
    if(l==r)
        return l;
    int mid=l+(r-l>>1);
    int sum=p[p[y].l].sum-p[p[x].l].sum;
    if(sum>=k) return query(l,mid,p[x].l,p[y].l,k);
    else return query(mid+1,r,p[x].r,p[y].r,k-sum);

}

int a[maxn];
vector <int >v;

int getid(int x)
{
    return lower_bound(v.begin(),v.end(),x)-v.begin()+1;
}

int main()
{
    int t;
    cin>>t;
    while(t--)
    {
        v.clear();
        tot=0;clr(root,0);
        cin>>n>>m;
        root[0]=build(1,n);
        for(int i=1; i<=n; i++)
        {
            cin>>a[i];
            v.push_back(a[i]);
        }
        sort(v.begin(),v.end());
        v.erase(unique(v.begin(),v.end()),v.end());
        for(int i=1; i<=n; i++)
            root[i]=update(1,n,root[i-1],getid(a[i]));
        int c,d,q;
        for(int i=1; i<=m; i++)
        {
            cin>>c>>d>>q;
            cout<<v[query(1,n,root[c-1],root[d],q)-1]<<endl;
        }
    }
    return 0;
}
```

### 12、KMP

```c++
#define ll long long
#define db double
#define clr(a,b) memset(a,b,sizeof(a))

int next[1000000+5];
void getnext(char *p)
{
    int i=0,j=-1;
    next[0]=-1;
    int m=strlen(p);
    while(i<m)
    {
        if(p[i]==p[j] || j==-1)
        {
            i++;j++;
            next[i]=j;
            if (p[i] != p[j])
				next[i] = j;
			else
				next[i] = next[i];
        }
        else
            j=next[j];
    }
}
int kmp(char* T,char* P)
{
    int res=0;
    int j=0,i=0;
    int n=strlen(T),m=strlen(P);
    while (i < n)
	{
		if (j == -1 || P[j] == T[i])
		{
			i++;
			j++;
		}
		else
			j = next[j];
		if (j == m)
		{
			//res++;
			//j = next[j];
			return i-m;
		}
	}
    //return res;
    return -1;
}

int n,len;
char p[10000+5],t[1000000+5];
int main()
{
    scanf("%d",&n);
    while(n--)
    {
        scanf("%s",p);
        scanf("%s",t);
        getnext(p);
        printf("%d\n",kmp(t,p));
    }

    return 0;
}
/*
Next数组的性质：
（1）如果 len-next[len] 能被 len 整除则 len - next[len] 是该串的循环节
（2）s[0] ~ s[next[len]-1] 中的内容一定能与 s[len-next[len]] ~ s[len-1] 匹配
（3）s[0] ~ s[next[i]-1] 中的内容一定能与 s[i-next[i]] ~ s[i-1] 匹配
*/
```

### 13、AC自动机 

```c++
int n;
char p[maxn];

struct Aho_Corasick
{
    int next[max_tot][26], nd[max_tot], fail[max_tot], vis[max_tot];
    int root, L;
    int newnode()
    {
        for(int i = 0; i < 26; i++)
            next[L][i] = -1;
        nd[L] = 0;
        vis[L] = 0;
        return L++;
    }
    void init()
    {
        L = 0;
        root = newnode();
    }
    void insert(char *s)
    {
        int now = root;
        for(int i = 0, key, sz = strlen(s); i < sz; i++)
        {
            key = s[i] - 'a';
            if(next[now][key] == -1)
                next[now][key] = newnode();
            now = next[now][key];
        }
        nd[now]++;
    }
    void build()
    {
        // fail数组含义
        // 和i节点代表的前缀的后缀匹配的trie上最长真前缀，由trie树性质得唯一
        // 即当i节点的某边发生失配时转移到达的trie上最长真前缀
        // if(next[i][j] == -1) next[i][j] = next[fail[i]][j]
        queue<int> Q;
        fail[root] = root;
        for(int i = 0; i < 26; i++)
        if(next[root][i] == -1)
            next[root][i] = root;
        else
        {
            fail[next[root][i]] = root;
            Q.push(next[root][i]);
        }

        while(!Q.empty())
        {
            int now = Q.front();
            Q.pop();
            for(int i = 0; i < 26; i++)
            if(next[now][i] == -1)
                next[now][i] = next[fail[now]][i];
            else
            {
                fail[next[now][i]] = next[fail[now]][i];
                Q.push(next[now][i]);
            }
        }
    }
    int query(char *s)
    {
        int now = root, ret = 0;
        for(int i = 0, key, tmp, sz = strlen(s); i < sz; i++)
        {
            key = s[i] - 'a';
            tmp = now = next[now][key];
            while(tmp != root && !vis[tmp])
            {
                ret += nd[tmp];
                nd[tmp] = 0;
                vis[tmp] = 1;
                tmp = fail[tmp];
            }
        }
        return ret;
    }
} aho;


int cas;
int main()
{
    scanf("%d",&cas);
    while(cas--)
    {
        aho.init();
        scanf("%d", &n);
        for( int i = 1; i <= n; i++)
        {
            scanf("%s", p);
            aho.insert(p);
        }
        aho.build();
        scanf("%s",p);
        printf("%d\n", aho.query(p));
    }
    return 0;
}
```

### 14、Dijkstra

```c++
#define ll long long
#define clr(a,b) memset(a,b,sizeof(a))
typedef pair<int ,int>  pii;
const int maxn=2e5+10;
priority_queue<pii,vector<pii>,greater<pii> >pq;
struct edge
{
    int to;
    int cost;
};
vector<edge> G[maxn];//g[i]--i to g[i].to cost cost
int n, m, s;
int dis[maxn];
void dijk(int s)
{
    for(int i = 1; i <= n; i++)
        dis[i] = inf;
    dis[s] = 0;
    pq.push(make_pair(0,s));
    while(!pq.empty())
    {
        pii u = pq.top();
        pq.pop();
        int x = u.second; // bian hao
        for(int i = 0; i < G[x].size(); i++)
        {
            edge e = G[x][i];
            if(dis[e.to] > dis[x] + e.cost)
            {
                dis[e.to] = dis[x] + e.cost;
                pq.push(make_pair(dis[e.to], e.to));
            }
        }
    }
}
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    cin >> n >> m >> s;
    int from, to, cost;
    edge in;
    for(int i = 0; i < m; i++)
    {
        cin >> from >> to >> cost;
        in.to = to; in.cost = cost;
        G[from].push_back(in);
    }
    dijk(s);
    for(int i = 1; i <= n; i++)
        cout << dis[i] << " ";
    return 0;
}
```

### 15、manacher

```c++

void manacher(string& s, int *R, int n)
{
/*
 * manacher算法
 * 需要将字符串预处理成$#x#x#x#x#x#x#形式 ✔
 * 若仅求长度为奇数的回文串，最左侧添加特殊字符即可
 * 记录当前最右延伸回文半径mx和对应回文中心p
 * i若位于mx以内，则将对称位置2*p-i的回文半径的不越界部分作为i的回文半径，并且继续向右侧匹配
 * 若得到新的最右延伸回文半径，更新mx和p
 * 回文长度为回文半径-1
 * 回文起始位置为 （回文中心位置-回文半径）/2
 */
    int p = 0, mx = 0;
    R[0] = 1;
    for(int i = 1; i < n; i++)
    {
        if(mx > i)
            R[i] = min(R[2*p - i], mx - i);
        else R[i] = 1;
        while(s[i - R[i]] == s[i + R[i]])
            R[i]++;
        if(i + R[i] > mx)
            p = i, mx = i + R[i];
    }
    return;
}
```

### 16、Palindromic_Tree

```c++
const int maxn = 100005;
const int N = 26;

struct Palindromic_Tree
{
	int next[maxn][N]; //next指针，next指针和字典树类似，指向的串为当前串两端加上同一个字符构成
	int fail[maxn];	//fail指针，失配后跳转到fail指针指向的节点
	int cnt[maxn], id[maxn];
	int num[maxn];
	int len[maxn]; //len[i]表示节点i表示的回文串的长度
	int S[maxn];   //存放添加的字符
	int last;	  //指向上一个字符所在的节点，方便下一次add
	int n;		   //字符数组指针
	int p;		   //节点指针

	int newnode(int l)
	{ //新建节点
		for (int i = 0; i < N; ++i)
			next[p][i] = 0;
		cnt[p] = 0;
		id[p] = 0;
		num[p] = 0;
		len[p] = l;
		return p++;
	}

	void init()
	{ //初始化
		p = 0;
		newnode(0);
		newnode(-1);
		last = 0;
		n = 0;
		S[n] = -1; //开头放一个字符集中没有的字符，减少特判
		fail[0] = 1;
	}

	int get_fail(int x)
	{ //和KMP一样，失配后找一个尽量最长的
		while (S[n - len[x] - 1] != S[n])
			x = fail[x];
		return x;
	}

	void add(int c)
	{
		c -= 'a';
		S[++n] = c;
		int cur = get_fail(last); //通过上一个回文串找这个回文串的匹配位置
		if (!next[cur][c])
		{											  //如果这个回文串没有出现过，说明出现了一个新的本质不同的回文串
			int now = newnode(len[cur] + 2);		  //新建节点
			fail[now] = next[get_fail(fail[cur])][c]; //和AC自动机一样建立fail指针，以便失配后跳转
			next[cur][c] = now;
			num[now] = num[fail[now]] + 1;
		}
		last = next[cur][c];
		cnt[last]++;
		id[last] = n; //id[第last个节点]=第n个字符
	}
	void insert(char *s)
	{
		int le = strlen(s), tmp;
		for (int i = 0; i < le; i++)
		{
			tmp = s[i] - 'a';
			add(tmp);
		}
	}
	
	void count()
	{
		for (int i = p - 1; i >= 0; --i)
			cnt[fail[i]] += cnt[i];
		//父亲累加儿子的cnt，因为如果fail[v]=u，则u一定是v的子回文串！
	}
};
```

### 17、dinic

```c++
#include<cstdio>
#include<cstring>
#include<queue>
#define inf 1e9
using namespace std;
const int maxn=500+5;
 
struct Edge
{
    int from,to,cap,flow;
    Edge(){}
    Edge(int f,int t,int c,int flow):from(f),to(t),cap(c),flow(flow){}
};
 
struct Dinic
{
    int n,m,s,t;
    vector<Edge> edges;
    vector<int> G[maxn];
    bool vis[maxn];
    int cur[maxn];
    int d[maxn];
 
    void init(int n,int s,int t)
    {
        this->n=n, this->s=s, this->t=t;
        edges.clear();
        for(int i=1;i<=n;i++) G[i].clear();
    }
 
    void AddEdge(int from,int to,int cap)
    {
        edges.push_back(Edge(from,to,cap,0));
        edges.push_back(Edge(to,from,0,0));
        m = edges.size();
        G[from].push_back(m-2);
        G[to].push_back(m-1);
    }
 
    bool BFS()
    {
        memset(vis,0,sizeof(vis));
        queue<int> Q;
        d[s]=0;
        Q.push(s);
        vis[s]=true;
        while(!Q.empty())
        {
            int x=Q.front(); Q.pop();
            for(int i=0;i<G[x].size();i++)
            {
                Edge& e=edges[G[x][i]];
                if(!vis[e.to] && e.cap>e.flow)
                {
                    vis[e.to]=true;
                    Q.push(e.to);
                    d[e.to]= 1+d[x];
                }
            }
        }
        return vis[t];
    }
 
    int DFS(int x,int a)
    {
        if(x==t || a==0) return a;
        int flow=0,f;
        for(int& i=cur[x]; i<G[x].size(); i++)
        {
            Edge& e=edges[G[x][i]];
            if(d[x]+1==d[e.to] && (f=DFS(e.to,min(a,e.cap-e.flow) ))>0 )
            {
                e.flow+=f;
                edges[G[x][i]^1].flow -=f;
                flow+=f;
                a-=f;
                if(a==0) break;
            }
        }
        return flow;
    }
 
    int dinic()
    {
        int flow=0;
        while(BFS())
        {
            memset(cur,0,sizeof(cur));
            flow += DFS(s,inf);
        }
        return flow;
    }
    
}DC;
 
int main()
{
    int n,m;
    while(scanf("%d%d",&m,&n)==2)
    {
        DC.init(n,1,n);
        while(m--)
        {
            int u,v,w;
            scanf("%d%d%d",&u,&v,&w);
            DC.AddEdge(u,v,w);
        }
        printf("%d\n",DC.dinic());
    }
    return 0;
}
```

### 18、ISAP

```c++
#define N 1000
#define INF 100000000
struct Edge
{
    int from,to,cap,flow;
};
struct ISAP
{
    int n,m,s,t;
    vector<Edge>edges;
    vector<int>G[N];
    bool vis[N];
    int d[N],cur[N];
    int p[N],num[N];//比Dinic算法多了这两个数组，p数组标记父亲结点，num数组标记距离d[i]存在几个
    void init()
    {
        for (int i = 0; i <= n;i++)
            G[i].clear();
        edges.clear();
         for (int i = 0; i <= n;i++)
         {
             d[i] = cur[i] = p[i] = num[i] = 0;
         }
    }
    
    void addedge(int from,int to,int cap)
    {
        edges.push_back((Edge){from,to,cap,0});
        edges.push_back((Edge){to,from,0,0});
        int m=edges.size();
        G[from].push_back(m-2);
        G[to].push_back(m-1);
    }
 
    int Augumemt()
    {
        int x=t,a=INF;
        while(x!=s)//找最小的残量值
        {
            Edge&e=edges[p[x]];
            a=min(a,e.cap-e.flow);
            x=edges[p[x]].from;
        }
        x=t;
        while(x!=s)//增广
        {
            edges[p[x]].flow+=a;
            edges[p[x]^1].flow-=a;
            x=edges[p[x]].from;
        }
        return a;
    }
    void bfs()//逆向进行bfs
    {
        memset(vis,0,sizeof(vis));
        queue<int>q;
        q.push(t);
        d[t]=0;
        vis[t]=1;
        while(!q.empty())
        {
            int x=q.front();q.pop();
            int len=G[x].size();
            for(int i=0;i<len;i++)
            {
                Edge&e=edges[G[x][i]];
                if(!vis[e.from]&&e.cap>e.flow)
                {
                    vis[e.from]=1;
                    d[e.from]=d[x]+1;
                    q.push(e.from);
                }
            }
        }
    }
 
    int Maxflow(int s,int t)//根据情况前进或者后退，走到汇点时增广
    {
        this->s=s;
        this->t=t;
        int flow=0;
        bfs();
        memset(num,0,sizeof(num));
        for(int i=0;i<n;i++)
            num[d[i]]++;
        int x=s;
        memset(cur,0,sizeof(cur));
        while(d[s]<n)
        {
            if(x==t)//走到了汇点，进行增广
            {
                flow+=Augumemt();
                x=s;//增广后回到源点
            }
            int ok=0;
            for(int i=cur[x];i<G[x].size();i++)
            {
                Edge&e=edges[G[x][i]];
                if(e.cap>e.flow&&d[x]==d[e.to]+1)
                {
                    ok=1;
                    p[e.to]=G[x][i];//记录来的时候走的边，即父边
                    cur[x]=i;
                    x=e.to;//前进
                    break;
                }
            }
            if(!ok)//走不动了，撤退
            {
                int m=n-1;//如果没有弧，那么m+1就是n，即d[i]=n
                for(int i=0;i<G[x].size();i++)
                {
                    Edge&e=edges[G[x][i]];
                    if(e.cap>e.flow)
                        m=min(m,d[e.to]);
                }
                if(--num[d[x]]==0)break;//如果走不动了，且这个距离值原来只有一个，那么s-t不连通，这就是所谓的“gap优化”
                num[d[x]=m+1]++;
                cur[x]=0;
                if(x!=s)
                    x=edges[p[x]].from;//退一步，沿着父边返回
            }
        }
        return flow;
    }
};
int main()
{
//    freopen("t.txt","r",stdin);
    int n,m;
    ISAP sap;
    while(cin>>n>>m)
    {
        sap.init();
        sap.n=n;
        sap.m = m;
        for(int i=0;i<sap.n;i++)
        {
            int from,to,cap;
            cin>>from>>to>>cap;
            sap.addedge(from,to,cap);
        }
       // cin>>sap.s>>sap.t;
        cout<<sap.Maxflow(1,sap.m)<<endl;
    }
    return 0;
}
 
 
```

### 19、大数

```c++
#include <iostream>
#include <cstdio>
#include <cstring>
#include <algorithm>
#include <vector>
#include <cmath>
#include <queue>

using namespace std;


const int maxn = 1000;

struct bign
{
    int d[maxn], len;
    void clean()
    {
        while(len > 1 && !d[len-1]) len--;    
    }
    bign()
    {
        memset(d, 0, sizeof(d));    
        len = 1;
    }
    bign(int num)
    {
        *this = num;
    }
    bign(char* num)
    {
        *this = num;
    }
    bign operator = (const char* num)
    {
        memset(d, 0, sizeof(d));
        len = strlen(num);
        for(int i = 0; i < len; i++) d[i] = num[len-1-i] - '0';
        clean();
        return *this;
    }
    bign operator = (int num)
    {
        char s[20];
        sprintf(s, "%d", num);
        *this = s;
        return *this;
}

    bign operator + (const bign& b)
    {
        bign c = *this;
        int i;
        for (i = 0; i < b.len; i++)
        {
            c.d[i] += b.d[i];
            if (c.d[i] > 9) c.d[i]%=10, c.d[i+1]++;
        }
        while (c.d[i] > 9) c.d[i++]%=10, c.d[i]++;
        c.len = max(len, b.len);
        if (c.d[i] && c.len <= i) c.len = i+1;
        return c;
    }
    bign operator - (const bign& b)
    {
        bign c = *this;
        int i;
        for (i = 0; i < b.len; i++)
        {
            c.d[i] -= b.d[i];
            if (c.d[i] < 0) c.d[i]+=10, c.d[i+1]--;
        }
        while (c.d[i] < 0) c.d[i++]+=10, c.d[i]--;
        c.clean();
        return c;
    }
    bign operator * (const bign& b)const
    {
        int i, j;
        bign c;
        c.len = len + b.len;
        for(j = 0; j < b.len; j++) for(i = 0; i < len; i++)
                c.d[i+j] += d[i] * b.d[j];
        for(i = 0; i < c.len-1; i++)
            c.d[i+1] += c.d[i]/10, c.d[i] %= 10;
        c.clean();
        return c;
    }
    bign operator / (const bign& b)
    {
        int i, j;
        bign c = *this, a = 0;
        for (i = len - 1; i >= 0; i--)
        {
            a = a*10 + d[i];
            for (j = 0; j < 10; j++) if (a < b*(j+1)) break;
            c.d[i] = j;
            a = a - b*j;
        }
        c.clean();
        return c;
    }
    bign operator % (const bign& b)
    {
        int i, j;
        bign a = 0;
        for (i = len - 1; i >= 0; i--)
        {
            a = a*10 + d[i];
            for (j = 0; j < 10; j++) if (a < b*(j+1)) break;
            a = a - b*j;
        }
        return a;
    }
    bign operator += (const bign& b)
    {
        *this = *this + b;
        return *this;
    }

    bool operator <(const bign& b) const
    {
        if(len != b.len) return len < b.len;
        for(int i = len-1; i >= 0; i--)
            if(d[i] != b.d[i]) return d[i] < b.d[i];
        return false;
    }
    bool operator >(const bign& b) const
    {
        return b < *this;
    }
    bool operator<=(const bign& b) const
    {
        return !(b < *this);
    }
    bool operator>=(const bign& b) const
    {
        return !(*this < b);
    }
    bool operator!=(const bign& b) const
    {
        return b < *this || *this < b;
    }
    bool operator==(const bign& b) const
    {
        return !(b < *this) && !(b > *this);
    }
    string str() const
    {
        char s[maxn]= {};
        for(int i = 0; i < len; i++) s[len-1-i] = d[i]+'0';
        return s;
    }
};
istream& operator >> (istream& in, bign& x)
{
    string s;
    in >> s;
    x = s.c_str();
    return in;
}
ostream& operator << (ostream& out, const bign& x)
{
    out << x.str();
    return out;
} 
int main()
{
    ios::sync_with_stdio(false);
    cin.tie(0);
    bign sum=0,t;
    while(cin >> t)
    {
        if(t.len==1&&!t.d[0]) break;
        sum+=t;
    }
    cout<<sum<<endl;
    return 0;
}
```

