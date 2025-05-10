#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/flann.hpp>

using namespace std;
using namespace cv;

// 图像数据结构
struct ImageData
{
    string path;
    vector<float> bow;
};

// 特征提取函数
vector<Mat> extractFeatures(const vector<string> &image_pathss, const string &feature_type = "SIFT")
{
    vector<Mat> all_descriptors;
    Ptr<Feature2D> feature_extractor;

    if (feature_type == "SIFT")
    {
        feature_extractor = SIFT::create();
    }
    else
    { // ORB
        feature_extractor = ORB::create();
    }

    for (const auto &path : image_pathss)
    {
        Mat img = imread(path, IMREAD_GRAYSCALE);
        if (img.empty())
        {
            cerr << "Warning: Could not read image " << path << endl;
            continue;
        }

        vector<KeyPoint> keypoints;
        Mat descriptors;
        feature_extractor->detectAndCompute(img, noArray(), keypoints, descriptors);

        if (!descriptors.empty())
        {
            all_descriptors.push_back(descriptors);
        }
    }

    return all_descriptors;
}

// 层次聚类构建视觉词典
Mat buildVocabulary(const vector<Mat> &all_descriptors, int K)
{
    // 检查K值
    if (K <= 0)
    {
        cerr << "K值无效: K=" << K << " (必须>0)" << endl;
        return Mat();
    }

    // 合并描述符
    Mat all_features;
    for (const auto &desc : all_descriptors)
    {
        if (desc.empty())
        {
            cerr << "警告: 遇到空描述符矩阵，跳过" << endl;
            continue;
        }
        Mat desc_float;
        desc.convertTo(desc_float, CV_32F); // 强制类型转换
        all_features.push_back(desc_float);
    }

    // 检查最终数据
    if (all_features.empty())
    {
        cerr << "错误: 所有描述符均为空或无效" << endl;
        cerr << "可能原因: " << endl;
        cerr << "1. 图片路径错误" << endl;
        cerr << "2. 图片无法被OpenCV解码" << endl;
        cerr << "3. 特征提取失败（如图片为纯色）" << endl;
        return Mat();
    }

    cout << "聚类输入数据维度: " << all_features.rows << "x" << all_features.cols
         << " (类型: " << all_features.type() << ")" << endl;

    // 执行kmeans
    Mat labels, centers;
    kmeans(all_features, K, labels,
           TermCriteria(TermCriteria::EPS + TermCriteria::MAX_ITER, 100, 1.0),
           3, KMEANS_PP_CENTERS, centers);
    return centers;
}

// 生成词袋向量
vector<float> generateBOW(const Mat &descriptors, const Mat &vocabulary, int K)
{
    // 使用FLANN进行最近邻搜索
    flann::KDTreeIndexParams indexParams;
    flann::Index kdtree(vocabulary, indexParams);

    vector<float> hist(K, 0.0f);
    Mat indices, dists;

    // 对每个描述符找到最近的视觉单词
    for (int i = 0; i < descriptors.rows; ++i)
    {
        Mat query = descriptors.row(i);
        kdtree.knnSearch(query, indices, dists, 1);

        int word_idx = indices.at<int>(0, 0);
        hist[word_idx]++;
    }

    // 归一化
    float norm = 0.0f;
    for (auto &val : hist)
        norm += val * val;
    norm = sqrt(norm);

    if (norm > 0)
    {
        for (auto &val : hist)
            val /= norm;
    }

    return hist;
}
vector<string> image_paths;
// 图像检索主函数
vector<pair<string, float>> imageRetrieval(
    const string &dataset_path,
    const string &query_path,
    int K,
    int top_k)
{

    // 1. 获取数据集所有图像路径 (需要实际实现)

    // 这里应该添加代码来遍历dataset_path目录下的所有图像文件
    // 例如使用boost::filesystem或dirent.h

    // 2. 特征提取
    auto all_descriptors = extractFeatures(image_paths);

    // 3. 构建视觉词典
    Mat vocabulary = buildVocabulary(all_descriptors, K);

    // 4. 为数据库图像生成词袋向量
    vector<ImageData> database;
    for (size_t i = 0; i < image_paths.size(); ++i)
    {
        ImageData data;
        data.path = image_paths[i];
        data.bow = generateBOW(all_descriptors[i], vocabulary, K);
        database.push_back(data);
    }

    // 5. 处理查询图像
    Mat query_img = imread(query_path, IMREAD_GRAYSCALE);
    if (query_img.empty())
    {
        cerr << "Error: Could not read query image " << query_path << endl;
        return {};
    }

    Ptr<Feature2D> sift = SIFT::create();
    vector<KeyPoint> query_kpts;
    Mat query_desc;
    sift->detectAndCompute(query_img, noArray(), query_kpts, query_desc);

    if (query_desc.empty())
    {
        cerr << "Error: Could not extract features from query image" << endl;
        return {};
    }

    vector<float> query_bow = generateBOW(query_desc, vocabulary, K);

    // 6. 计算相似度并排序
    vector<pair<string, float>> results;
    for (const auto &img_data : database)
    {
        float similarity = 0.0f;
        for (size_t i = 0; i < K; ++i)
        {
            similarity += query_bow[i] * img_data.bow[i];
        }
        results.emplace_back(img_data.path, similarity);
    }

    // 按相似度降序排序
    sort(results.begin(), results.end(),
         [](const pair<string, float> &a, const pair<string, float> &b)
         {
             return a.second > b.second;
         });

    // 返回top_k结果
    if (results.size() > top_k)
    {
        results.resize(top_k);
    }

    return results;
}

int main()
{
    string dataset_path = "./photo/photo";
    string query_path = "./photo/photo/7.png";
    int K = 100;   // 词典大小
    int top_k = 5; // 返回结果数量
                   // vector<string> image_paths;
    for (int i = 1; i <= 300; ++i)
    {
        image_paths.push_back(dataset_path + "/" + to_string(i) + ".png");
    }
    auto results = imageRetrieval(dataset_path, query_path, K, top_k);

    // 输出结果
    cout << "Top " << top_k << " similar images:" << endl;
    for (const auto &result : results)
    {
        cout << "Image: " << result.first << " - Similarity: " << result.second << endl;
    }

    return 0;
}