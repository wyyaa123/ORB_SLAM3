#include "edgeExtracter.h"

std::vector<BezierCurve> edgeExtracter::operator()(cv::Mat imGray, cv::Mat imSem)
{
    // Edge extraction and processing
    cv::Canny(imGray, mCanny, threshold1, threshold2, 3, true);

    cv::Mat grad_x, grad_y;
    cv::Scharr(imGray, grad_x, CV_32F, 1, 0);
    cv::Scharr(imGray, grad_y, CV_32F, 0, 1);

    mMatGradMagnitude.create(imGray.size(), CV_32F);
    mMatGradAngle.create(imGray.size(), CV_32F);

    cv::cartToPolar(grad_x, grad_y, mMatGradMagnitude, mMatGradAngle, true);

    preprocessSem(imSem);

    preprocessEdge();

    regionGrowthClusteringOCanny(angleThreshold);

    cvt2OrderedEdges();

    getBezierSampledPoints(3);

    return mvBezierCurves;
}

inline float edgeExtracter::calcAngleBias(float angle_1, float angle_2)
{
    float res = fabs(angle_1 - angle_2);
    if (res > 180)
    {
        res = 360 - res;
    }
    return res;
}

void edgeExtracter::preprocessSem(const cv::Mat &imSem)
{
    if (imSem.empty())
    {
        mSem = cv::Mat::ones(mCanny.size(), CV_8UC1);
        return;
    }

    cv::Mat filteredSem = cv::Mat::zeros(imSem.size(), CV_8UC1);
    for (int i = 1; i < 10; ++i)
    {
        cv::Mat temp = cv::Mat::zeros(imSem.size(), CV_8UC1);
        temp.setTo(255, imSem == i);

        if (cv::countNonZero(temp) == 0)
            continue;

        cv::Mat label, stats, centroids;
        cv::connectedComponentsWithStats(temp, label, stats, centroids, 4, CV_16U);

        for (int j = 1; j < centroids.rows; j++)
        {
            int area = stats.at<int>(j, cv::CC_STAT_AREA);
            int top = stats.at<int>(j, cv::CC_STAT_TOP);
            int left = stats.at<int>(j, cv::CC_STAT_LEFT);
            int width = stats.at<int>(j, cv::CC_STAT_WIDTH);
            int height = stats.at<int>(j, cv::CC_STAT_HEIGHT);

            if (area < 100)
                continue;

            double WHratio = std::max(width, height) / std::min(width, height);
            if (WHratio > 10)
                continue;

            cv::Mat mask = cv::Mat::zeros(imSem.size(), CV_8UC1);
            mask = mask.setTo(255, label == j);

            cv::Mat close_kernel = cv::Mat::ones(5, 5, CV_8UC1);
            cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, close_kernel);

            cv::Mat open_kernel = cv::Mat::ones(5, 5, CV_8UC1);
            cv::morphologyEx(mask, mask, cv::MORPH_OPEN, open_kernel);

            int dilationSize = 2;
            cv::Mat dilationElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2 * dilationSize + 1, 2 * dilationSize + 1), cv::Point(dilationSize, dilationSize));
            cv::dilate(mask, mask, dilationElement);

            filteredSem.setTo(i, mask);
        }
    }
    mSem = filteredSem;
}

void edgeExtracter::preprocessEdge()
{
    for (int i = 0; i < mCanny.rows; ++i)
    {
        uint8_t *current = mCanny.ptr<uint8_t>(i);
        uint8_t *above = i > 0 ? mCanny.ptr<uint8_t>(i - 1) : nullptr;
        uint8_t *below = i < mCanny.rows - 1 ? mCanny.ptr<uint8_t>(i + 1) : nullptr;

        for (int j = 0; j < mCanny.cols; ++j)
        {
            if (current[j] == 0)
                continue; // 跳过非边缘点

            int left = j > 0 ? current[j - 1] : 0;
            int right = j < mCanny.cols - 1 ? current[j + 1] : 0;
            int up = above ? above[j] : 0;
            int down = below ? below[j] : 0;

            bool connected = (left > 0 && up > 0) || (right > 0 && up > 0) ||
                             (left > 0 && down > 0) || (right > 0 && down > 0);

            if (connected)
            {
                current[j] = 0;
            }
        }
    }
}

void edgeExtracter::cvt2OrderedEdges()
{
    mvEdges.reserve(mvEdgeClusters.size());

    const float *angle_ptr = mMatGradAngle.ptr<float>(0);
    const int angle_step = mMatGradAngle.step / sizeof(float);

    for (int i = 0; i < mvEdgeClusters.size(); ++i)
    {
        Edge curr_edge(i);
        const auto &cluster_points = mvEdgeClusters[i].organize();
        ;
        curr_edge.mvPoints.reserve(cluster_points.size());

        for (const auto &point : cluster_points)
        {
            const int x = static_cast<int>(point.pixel.x);
            const int y = static_cast<int>(point.pixel.y);

            // angle(0~360) of the image gradient
            const float angle = angle_ptr[y * angle_step + x];

            // an orderedEdgePoint is initially constructed by coordinate (x,y) and gradient angle
            curr_edge.mvPoints.emplace_back(x, y, angle);
        }
        curr_edge.cls = cluster_points.front().cls; // 直接取第一个点的语义类别作为边缘的类别
        mvEdges.push_back(std::move(curr_edge));
    }
}

void edgeExtracter::regionGrowthClusteringOCanny(float angle_Thres, const cv::Point &offset)
{
    cv::Mat visitedMat(mCanny.rows, mCanny.cols, CV_8UC1, cv::Scalar::all(0));
    mvEdgeClusters.clear();

    uint8_t *canny_ptr = mCanny.data;
    uint8_t *sem_ptr = mSem.data;
    uint8_t *visited_ptr = visitedMat.data;
    const float *angle_ptr = mMatGradAngle.ptr<float>();
    const int canny_step = static_cast<int>(mCanny.step);
    const int sem_step = static_cast<int>(mSem.step);
    const int visited_step = static_cast<int>(visitedMat.step);
    const int angle_step = static_cast<int>(mMatGradAngle.step / sizeof(float));
    const int width = mCanny.cols;
    const int height = mCanny.rows;

    static const int kDx[8] = {1, 1, 0, -1, -1, -1, 0, 1};
    static const int kDy[8] = {0, 1, 1, 1, 0, -1, -1, -1};

    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            if (visited_ptr[y * visited_step + x] != 0 || canny_ptr[y * canny_step + x] != 255 || sem_ptr[y * sem_step + x] == 0)
                continue;

            std::vector<edgePoint> current_cluster;
            current_cluster.reserve(128);
            visited_ptr[y * visited_step + x] = 1;
            int point_id = 0;
            int cls = sem_ptr[y * sem_step + x];

            edgePoint curr_edge_point(cv::Point(x, y), point_id, -1, cls);
            curr_edge_point.isRoot = true;
            point_id++;

            std::vector<edgePoint> open_list;
            open_list.reserve(256);
            open_list.push_back(curr_edge_point);
            size_t head = 0;

            while (head < open_list.size())
            {
                const edgePoint current_point = open_list[head++];

                const int cx = current_point.pixel.x;
                const int cy = current_point.pixel.y;

                current_cluster.push_back(current_point);

                const float curr_angle = angle_ptr[(cy + offset.y) * angle_step + (cx + offset.x)];

                for (int k = 0; k < 8; ++k)
                {
                    const int nx = cx + kDx[k];
                    const int ny = cy + kDy[k];
                    if (nx < 0 || nx >= width || ny < 0 || ny >= height || sem_ptr[ny * sem_step + nx] != cls)
                    {
                        continue;
                    }

                    const int visited_idx = ny * visited_step + nx;
                    if (visited_ptr[visited_idx] != 0)
                    {
                        continue;
                    }

                    if (canny_ptr[ny * canny_step + nx] != 255)
                    {
                        continue;
                    }

                    const float neigh_angle = angle_ptr[(ny + offset.y) * angle_step + (nx + offset.x)];

                    float angle_bias = calcAngleBias(neigh_angle, curr_angle);
                    if (angle_bias >= angle_Thres)
                    {
                        continue;
                    }

                    visited_ptr[visited_idx] = 1;
                    open_list.emplace_back(cv::Point(nx, ny), point_id, current_point.point_id, cls);
                    point_id++;
                }
            }

            if (current_cluster.size() > 10)
            {
                mvEdgeClusters.emplace_back(std::move(current_cluster));
            }
        }
    }
}

void edgeExtracter::getFineSampledPoints(int bias)
{
    for (int i = 0; i < mvEdges.size(); ++i)
    {
        Edge &edge = mvEdges[i];
        edge.samplingEdgeUniform(bias);
    }
}

void edgeExtracter::getBezierSampledPoints(int sampleSpacing)
{
    const BezierCurveFitter bezierFitter(1);
    mvBezierCurves.clear();
    mvBezierCurves.reserve(mvEdges.size());
    for (auto &edge : mvEdges)
    {
        // 生成贝塞尔曲线控制点
        const std::vector<BezierCurve> fittedCurves = bezierFitter.fitAdaptive(edge.mvPoints);
        for (BezierCurve curve : fittedCurves)
        {
            if (curve.controlPoints.size() < 2)
                continue;

            curve.edge_ID = edge.edge_ID;
            curve.cls = edge.cls;

            curve.sampleByArcLengthSpacing(sampleSpacing);
            if (curve.sampledPoints.size() < 2)
                continue;

            mvBezierCurves.push_back(std::move(curve));
        }
    }
}