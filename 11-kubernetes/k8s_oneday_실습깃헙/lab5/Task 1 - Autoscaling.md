#  AutoScaling

# Metic-server 설치

```
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/download/v0.5.0/components.yaml
```

# Deployment 편집

```
kubectl edit deployment metrics-server -n kube-system
```

```
/kubelet-use-node-status-port

- --kubelet-use-node-status-port
- --metric-resolution=15s
- --kubelet-insecure-tls
```
# 맨 아랫줄 추가


# 동작 확인

```
kubectl top node
```

# 실습에 필요한 Deployment와 Service 생성

```
kubectl apply -f https://k8s.io/examples/application/php-apache.yaml
```

# HPA 생성

```
kubectl autoscale deployment php-apache --cpu-percent=50 --min=1 --max=10
```

# 확인

```
kubectl get hpa
```

# 부하 생성(다른 터미널을 추가하여 진행)

```
kubectl run -i --tty load-generator --rm --image=busybox --restart=Never -- /bin/sh -c "while sleep 0.01; do wget -q -O- http://php-apache; done"
```

# 1분정도 뒤 확인

```
kubectl get hpa
kubectl get deployment php-apache
```


# 부하 중지(부하 생성 시 터미널에서 진행)

```
Ctrl + C
```

# 확인

```
kubectl get hpa
kubectl get deployment php-apache
```

참고 : [https://kubernetes.io/ko/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/](https://kubernetes.io/ko/docs/tasks/run-application/horizontal-pod-autoscale-walkthrough/)
