# Task 2 - Pod.md  

### Pod를 직접 생성/조회/삭제
#
1. 실습 디렉토리 이동
```
cd ~/k8s_oneday_essential/lab2/yaml
```

2. Pod 조회
```
kubectl get pods
kubectl get pod
kubectl get po
```

3. yaml 확인
```
cat pod.yaml
```

4. yaml 을 활용한 Pod 생성
```
kubectl create -f pod.yaml
```

5. Pod 조회 및 세부정보 확인
```
kubectl get po
kubectl describe po
```

6. Pod 삭제
```
kubectl delete po <Pod명>
kubectl delete po --all
```

7. Pod 조회
```
kubectl get pod
```

8. run 명령을 사용한 pod 생성

>조건  
pod name : lab2-pod  
container image : nginx  
container port : 80
```
kubectl run lab2-pod --image=nginx --port=80
```

9. 8에서 생성한 pod 조회
```
kubectl get po
kubectl describe po
```

10. 8에서 생성한 pod 삭제
```
kubectl delete pod lab2-pod
kubectl delete pod --all
```
