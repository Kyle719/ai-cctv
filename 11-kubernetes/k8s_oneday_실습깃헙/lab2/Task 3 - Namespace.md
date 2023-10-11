# Task 3 - Namespace

### namespace 를 직접 생성/조회/삭제 및 namespace 안에 리소스 생성/조회/삭제
#  

1. Namespace 확인
```
kubectl get namespace
kubectl get ns
```

2. yaml 확인
```
cat ns1.yaml
```

3. yaml 을 활용한 namespace 생성
```
kubectl create -f ns1.yaml
```

4. Namespace 재확인
```
kubectl get ns
```

5. n1 namespace 안에 pod 생성
```
kubectl create -f pod.yaml -n ns1
```

6. pod 확인
```
kubectl get po
kubectl get po -n ns1
```

7. 5에서 생성한 Pod 삭제
```
kubectl delete pod --all -n ns1
```

8. pod 재확인
```
kubectl get pod -n ns1
```

9. 3에서 생성한 Namespace 삭제
```
kubectl delete ns ns1
```

10. create, run 명령으로 namespace 와 pod 생성
>-namespace 조건-  
namespace name : ns2  

> -pod 조건-  
pod name : ns-pod  
namespace : ns2  
container image : nginx  
container port : 80

```
kubectl create ns ns2
kubectl run ns-pod --image=nginx --port=80 -n ns2
```

11. 10에서 생성한 namespace와 pod 조회
```
kubectl get ns
kubectl get pod -n ns2
```

12. namespace 삭제로 해당 namespace에 있던 리소스 모두와 함께 삭제
```
kubectl delete ns ns2
```

13. 삭제 확인
```
kubectl get ns
```
