# Task 4 - Replicaset

### Replicaset 을 직접 생성/조회/삭제
#  

1. Replicaset 확인
```
kubectl get replicaset
kubectl get rs
```

2. yaml 확인
```
cat rs1.yaml
```

3. yaml 을 활용한 replicaset 생성
```
kubectl create -f rs1.yaml
```

4. replicaset, pod 확인
```
kubectl get rs
kubectl get pod
```

5. pod 1개 삭제
```
kubectl delete pod <pod명>
```

6. pod 재배포 확인
```
kubectl get po
```

7. replicaset 삭제
```
kubectl delete rs --all
```

8. replicaset 삭제 확인
```
kubectl get rs
kubectl get pod
```
