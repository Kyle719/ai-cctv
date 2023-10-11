# Task1 - Version, Resource.md  

### 2가지 명령을 진행함으로, 버전과 kubernetes에서 사용되는 리소스의 api 버전, 목록, short name 들을 확인
#
1. kubectl 버전확인
```
kubectl version
```

2. 리소스 목록, api 버전, short name 확인
```
kubectl api-resources
```  
![image](https://user-images.githubusercontent.com/92773629/138027152-bbde5c41-2a20-4516-b366-b036e47dcad2.png)  
3. 현재 적용되어있는 클러스터 확인
```
kubectl config current-context
```  

4. 클러스터의 유저와 정보 확인
```
kubectl config view
```

5. 실습 파일 다운로드
```
git clone https://github.com/wsjang619/k8s_oneday_essential
```
