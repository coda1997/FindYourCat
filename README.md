# 寻找我的猫猫

![cover](./img/cover.jpeg)


# 需求
## 场景1
用户想知道一个猫猫是什么品种的，直接打开小程序，点击拍照，然后上传照片到服务端，服务端会返回一个带有猫猫品种、性别、年龄的json文件。小程序根据这个json来给用户介绍这是什么猫。

## 具体步骤

### 小程序侧
1. 有一个UI，用户友好的
2. 点击拍摄图片
3. 上传图片
4. 接受服务器的返回结果
### 服务侧
1. 加载一个神经网络模型
2. 相应图片识别的请求
3. 接收图片
4. 模型识别
5. 返回结果

// TODO: 图片数据持久化，服务器预热，负载均衡，Authentication

# 技术栈
- Wechat mini app
- Service: Spring Boot
- AI related work: Pytorch, ported into java.


# 里程碑
1. End-to-end testing.

# Task
Client：
- [ ] Mini app onboard
- [ ] Send a request with a Img file attached.
- [ ] Receive a pre-defined json.
Service：
- [ ] Spring Boot onboard.
- [ ] Add endpoint like https://127.0.0.1:{port}/catbreed/uploading.
- [ ] Accept [POST] request from client.
- [ ] Return Json.

``` json
{
  “breed”: “cat”,
  “gender”: “male or female”,
  “age”: 3,
}
```
