# Task 5 - Deployment

### Deployment 을 생성, 조회, 업데이트, 롤백, 스케일링, 삭제
#  

1. Deployment 확인
```
kubectl get deployment
kubectl get deploy
```

2. yaml 확인
```
cat dp1.yaml
```

3. yaml 을 활용한 Deployment 생성
```
kubectl create -f dp1.yaml
```

4. deployment, replicaset, pod 확인
```
kubectl get deploy
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

7. Deployment 삭제
```
kubectl delete deploy --all
```

8. Deployment 삭제 확인
```
kubectl get deploy
kubectl get pod
```

9. 터미널을 하나 더 오픈하여 모니터링용 터미널을 생성합니다.
```
watch -n 0.5 kubectl get pod
```

10. create 명령으로 deployment 생성
>-deployment 조건-  
deployment name : dp2  
container image : nginx:1.14.0  
container port : 80  
replicas : 3  

```
kubectl create deploy dp2 --image=nginx:1.14.0 --port=80 --replicas=3
```


11. 컨테이너 이미지 1.15.0으로 버전 업데이트
```
kubectl set image deployment/dp2 nginx=nginx:1.15.0 --record=true
```
`이때 명령어 수행 직후 모니터링 터미널로 동작 확인`  
`record=true 값으로 해야 히스토리 확인시 어떤 내용인지 확인 가능`

12. 업데이트 내역 확인
```
kubectl describe pod
kubectl describe deploy
```

13. 업데이트 방식 변경
```
kubectl edit deploy dp2
```
```
strategy:
  rollingUpdate:
    maxSurge: 25%
    maxUnavailable: 25%
  type: RollingUpdate
```
를  
```
strategy:
  type: Recreate
```
로 수정합니다.

`vi 편집기 사용법과 동일합니다.`

14. 컨테이너 이미지 1.16.0 으로 버전 업데이트
```
kubectl set image deployment/dp2 nginx=nginx:1.16.0 --record=true
```
`이때 명령어 수행 직후 모니터링 터미널로 동작 확인`   

15. 업데이트 내역 확인
```
kubectl describe pod
kubectl describe deploy
```

16. 롤 아웃 기록 확인
```
kubectl rollout history deploy/dp2
```

17. 직전 버전으로 롤백
```
kubectl rollout undo deploy/dp2
```

18. 버전 확인
```
kubectl describe deploy
```

19. 리비전 지정하여 롤백
```
kubectl rollout undo deploy/dp2 --to-revision=1
```

20. 버전 확인
```
kubectl describe deploy
```

21. 스케일링
```
kubectl scale deploy/dp2 --replicas=5
```

22. 결과 확인
```
kubectl get pod
kubectl describe deploy
```

23. deployment 삭제
```
kubectl delete deploy --all
```
