Header header 

int32[] startRingIndex   # 每条线起始索引号
int32[] endRingIndex     # 每条线终止索引号

float32 startOrientation
float32 endOrientation
float32 orientationDiff

# 将16线，用一维数组存储
bool[]    segmentedCloudGroundFlag # true - ground point, false - other points，带标记，是否为地面的点
uint32[]  segmentedCloudColInd # point column index in range image， 此点的索引号
float32[] segmentedCloudRange # point range， 此点的距离 




