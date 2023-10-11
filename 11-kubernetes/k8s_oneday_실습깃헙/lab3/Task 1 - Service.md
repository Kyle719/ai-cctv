# Task1 - Service  

### 쿠버네티스의 통신을 담당하는 리소스인 Service를 생성, 조회, 삭제 및 기능확인
#  
1. 실습 디렉토리 이동
```
cd ~/k8s_oneday_essential/lab3/yaml
```

2. Service 조회
```
kubectl get service
kubectl get svc
```  
#
#### ClusterIP Type Service


3. yaml 확인

```
cat svc-pod.yaml
cat svc1.yaml
```  

4. yaml 을 활용한 ClusterIP 유형의 Service 생성
```
kubectl create -f svc1.yaml
```

5. yaml 을 활용한 Servivec와 연결할 Pod 생성
```
kubectl create -f svc-pod.yaml
```

6. 위 4~5 에서 만든 리소스 확인
```
kubectl get pod
kubectl get svc
```

7. curl 명령을 통한 통신 확인
```
curl <svc의 ClusterIP>:9090
```  
`6에서 확인한 IP를 입력합니다.`

#
#### Nodeport Type Service


8. 두번째 Servcie yaml 확인
```
cat svc2.yaml
```


9. yaml 을 활용한 Nodeport 유형의 Service 생성
```
kubectl create -f svc2.yaml
```

10. 생성된 Service 확인
```
kubectl get svc
```

11. 5에서 생성한 Pod가 어떤 노드에 생성되었는지 확인
```
kubectl describe pod
```


12. curl 명령을 통한 통신 확인
```
curl <Worker노드의 IP>:30000
```
`11에서 확인한 노드의 IP를 기입합니다.`

#
#### Loadbalancer Type Service

12. 세번째 Servcie yaml 확인
```
cat svc3.yaml
```
13. yaml 을 활용한 Loadbalancer 유형의 Service 생성
```
kubectl create -f svc3.yaml
```

14. 생성한 Service 확인
```
kubectl get svc
```
