#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <fstream>
#include <chrono>

using namespace std;
using namespace std::chrono;

// 点数据结构
struct Point
{
    vector<double> coords;
    int id;

    Point(const vector<double> &c, int i) : coords(c), id(i) {}
};

// KD树节点
struct KDNode
{
    Point point;
    shared_ptr<KDNode> left;
    shared_ptr<KDNode> right;
    int split_dim;

    KDNode(const Point &p) : point(p), left(nullptr), right(nullptr), split_dim(0) {}
};

// 优先队列元素
struct QueueItem
{
    shared_ptr<KDNode> node;
    double priority;

    QueueItem(shared_ptr<KDNode> n, double p) : node(n), priority(p) {}

    bool operator<(const QueueItem &other) const
    {
        return priority < other.priority;
    }
};

// KD树类
class KDTree
{
private:
    shared_ptr<KDNode> root;

    shared_ptr<KDNode> buildTree(vector<Point> &points, int depth)
    {
        if (points.empty())
            return nullptr;

        int k = points[0].coords.size();
        int axis = depth % k;

        // 按当前维度排序
        sort(points.begin(), points.end(), [axis](const Point &a, const Point &b)
             { return a.coords[axis] < b.coords[axis]; });

        int median = points.size() / 2;
        shared_ptr<KDNode> node = make_shared<KDNode>(points[median]);
        node->split_dim = axis;

        // 递归构建左右子树
        vector<Point> leftPoints(points.begin(), points.begin() + median);
        vector<Point> rightPoints(points.begin() + median + 1, points.end());

        node->left = buildTree(leftPoints, depth + 1);
        node->right = buildTree(rightPoints, depth + 1);

        return node;
    }

public:
    KDTree(vector<Point> &points)
    {
        root = buildTree(points, 0);
    }

    // 暴力搜索
    Point bruteForceSearch(const Point &query)
    {
        Point best = root->point;
        double best_dist = numeric_limits<double>::max();

        vector<shared_ptr<KDNode>> nodes;
        nodes.push_back(root);

        while (!nodes.empty())
        {
            auto current = nodes.back();
            nodes.pop_back();

            double dist = 0;
            for (int i = 0; i < query.coords.size(); ++i)
            {
                dist += pow(current->point.coords[i] - query.coords[i], 2);
            }
            dist = sqrt(dist);

            if (dist < best_dist)
            {
                best_dist = dist;
                best = current->point;
            }

            if (current->left)
                nodes.push_back(current->left);
            if (current->right)
                nodes.push_back(current->right);
        }

        return best;
    }

    // 标准KD树搜索
    Point kdTreeSearch(const Point &query)
    {
        Point best = root->point;
        double best_dist = numeric_limits<double>::max();
        searchKDTree(root, query, best, best_dist, 0);
        return best;
    }

    void searchKDTree(shared_ptr<KDNode> node, const Point &query, Point &best, double &best_dist, int depth)
    {
        if (!node)
            return;

        double dist = 0;
        for (int i = 0; i < query.coords.size(); ++i)
        {
            dist += pow(node->point.coords[i] - query.coords[i], 2);
        }
        dist = sqrt(dist);

        if (dist < best_dist)
        {
            best_dist = dist;
            best = node->point;
        }

        int axis = depth % query.coords.size();
        bool go_left = query.coords[axis] < node->point.coords[axis];

        if (go_left)
        {
            searchKDTree(node->left, query, best, best_dist, depth + 1);
            if (pow(query.coords[axis] - node->point.coords[axis], 2) < best_dist)
            {
                searchKDTree(node->right, query, best, best_dist, depth + 1);
            }
        }
        else
        {
            searchKDTree(node->right, query, best, best_dist, depth + 1);
            if (pow(query.coords[axis] - node->point.coords[axis], 2) < best_dist)
            {
                searchKDTree(node->left, query, best, best_dist, depth + 1);
            }
        }
    }

    // BBF搜索
    Point bbfSearch(const Point &query, int t)
    {
        priority_queue<QueueItem> queue;
        queue.emplace(root, numeric_limits<double>::max());

        Point best = root->point;
        double best_dist = numeric_limits<double>::max();
        int searched_nodes = 0;

        while (!queue.empty() && searched_nodes < t)
        {
            auto item = queue.top();
            queue.pop();
            auto node = item.node;

            if (!node->left && !node->right)
            {
                searched_nodes++;
                double dist = 0;
                for (int i = 0; i < query.coords.size(); ++i)
                {
                    dist += pow(node->point.coords[i] - query.coords[i], 2);
                }
                dist = sqrt(dist);

                if (dist < best_dist)
                {
                    best_dist = dist;
                    best = node->point;
                }
            }
            else
            {
                int dim = node->split_dim;
                double diff = query.coords[dim] - node->point.coords[dim];
                auto nearer = diff < 0 ? node->left : node->right;
                auto farther = diff < 0 ? node->right : node->left;

                if (farther)
                {
                    double priority = 1.0 / (diff * diff);
                    queue.emplace(farther, priority);
                }

                if (nearer)
                {
                    double priority = 1.0 / (diff * diff);
                    queue.emplace(nearer, priority);
                }
            }
        }

        return best;
    }
};

// 从文件读取数据
vector<Point> readDataFromFile(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "无法打开文件: " << filename << endl;
        exit(1);
    }

    int n, m, d;
    file >> n >> m >> d;

    vector<Point> data;
    for (int i = 0; i < n; ++i)
    {
        vector<double> coords(d);
        for (int j = 0; j < d; ++j)
        {
            file >> coords[j];
        }
        data.emplace_back(coords, i);
    }

    return data;
}

// 从文件读取查询点
vector<Point> readQueriesFromFile(const string &filename)
{
    ifstream file(filename);
    if (!file.is_open())
    {
        cerr << "无法打开文件: " << filename << endl;
        exit(1);
    }

    int n, m, d;
    file >> n >> m >> d;

    // 跳过数据点
    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < d; ++j)
        {
            double temp;
            file >> temp;
        }
    }

    vector<Point> queries;
    for (int i = 0; i < m; ++i)
    {
        vector<double> coords(d);
        for (int j = 0; j < d; ++j)
        {
            file >> coords[j];
        }
        queries.emplace_back(coords, i);
    }

    return queries;
}

// 测试函数
void runTests(const string &filename)
{
    // 读取数据
    auto data = readDataFromFile(filename);
    KDTree tree(data);

    // 读取查询点
    auto queries = readQueriesFromFile(filename);

    // 计算距离函数
    auto calc_dist = [](const Point &a, const Point &b)
    {
        double dist = 0;
        for (int i = 0; i < a.coords.size(); ++i)
        {
            dist += pow(a.coords[i] - b.coords[i], 2);
        }
        return sqrt(dist);
    };

    // 测试每个查询点
    for (size_t i = 0; i < min(queries.size(), static_cast<size_t>(100)); ++i)
    {
        const auto &query = queries[i];

        // 暴力搜索
        auto start = high_resolution_clock::now();
        auto bf_result = tree.bruteForceSearch(query);
        auto bf_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count();

        // KD树搜索
        start = high_resolution_clock::now();
        auto kd_result = tree.kdTreeSearch(query);
        auto kd_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count();

        // BBF搜索
        start = high_resolution_clock::now();
        auto bbf_result = tree.bbfSearch(query, 200);
        auto bbf_time = duration_cast<microseconds>(high_resolution_clock::now() - start).count();

        // 计算距离
        double bf_dist = calc_dist(query, bf_result);
        double kd_dist = calc_dist(query, kd_result);
        double bbf_dist = calc_dist(query, bbf_result);

        // 计算准确率
        double kd_accuracy = bf_dist > 0 ? bf_dist / kd_dist : 1.0;
        double bbf_accuracy = bf_dist > 0 ? bf_dist / bbf_dist : 1.0;

        // 输出结果
        cout << "查询点 " << i << ":" << endl;
        cout << "暴力搜索时间: " << bf_time << " μs" << endl;
        cout << "KD树搜索时间: " << kd_time << " μs, 准确率: " << kd_accuracy << endl;
        cout << "BBF搜索时间: " << bbf_time << " μs, 准确率: " << bbf_accuracy << endl;
        cout << "----------------------------------------" << endl;
    }
}

int main()
{
    setlocale(LC_ALL, "zh_CN.UTF-8");
    // 测试数据文件
    string filename = "./data/1.txt"; // 修改为实际文件路径
    runTests(filename);

    return 0;
}