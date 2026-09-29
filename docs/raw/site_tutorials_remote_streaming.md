# Remote Streaming - BEHAVIOR

> 원본: https://behavior.stanford.edu/tutorials/remote_streaming.html
> 받은 시각: 2026-09-29 18:22 KST (2026-09-29 09:22 UTC)
> 비고: 대회 페이지 본문이 링크하거나 대회 준비에 직접 쓰이는 사이트 문서
> 이 파일은 `_scrape.py` 가 자동으로 받은 원문 보관본이다. HTML→Markdown 변환 중 서식은 달라질 수 있으나 문장은 원문 그대로다.

---
# **Remote Streaming through WebRTC**

This page describes the steps required to set up and troubleshoot remote streaming for Omnigibson using WebRTC. The feature allows users to run Omnigibson on a remote server and view it locally through a browser.

## Enabling Remote Streaming

To enable remote streaming with WebRTC, you need to set an environment variable before launching Omnigibson. Use the following command in your terminal:

```
export OMNIGIBSON_REMOTE_STREAMING=webrtc
```

When you start Omnigibson, it will provide a URL that you can use to connect from your local machine.

## Configuring Ports

By default, Omnigibson uses two ports for remote streaming:

- HTTP Port (8211): This port serves the webpage used to connect via WebRTC.
- WebRTC Port (49100): This port handles the actual WebRTC connection.

## Changing the HTTP Port

To change the HTTP port, set the following environment variable to your desired port number:

```
export OMNIGIBSON_HTTP_PORT=<desired_port_number>
```

## Changing the WebRTC Port

To change the WebRTC port, set the following environment variable to your desired port number:

```
export OMNIGIBSON_WEBRTC_PORT=<desired_port_number>
```

Note: Changing the WebRTC port requires an additional step. You need to update the JavaScript file that the HTTP server serves to ensure it points to the new WebRTC port. Depending on your installation (Docker or local), use one of the following commands:

### For Docker:

```
sed -i "s/49100/${OMNIGIBSON_WEBRTC_PORT}/g" /isaac-sim/extscache/omni.services.streamclient.webrtc-1.3.8/web/js/kit-player.js
```

### For Local Installation:

```
sed -i "s/49100/${OMNIGIBSON_WEBRTC_PORT}/g" ~/.local/share/ov/pkg/isaac_sim-2023.1.1/web/js/kit-player.js
```

## Troubleshooting

### Issue: Undefined Symbol Error

When opening the remote streaming webpage, you might see an error related to libssl.so.1.1. If you see the following error:

```
python: symbol lookup error: /isaac-sim/extscache/omni.kit.streamsdk.plugins-2.5.2+105.1.lx64.r/bin/libssl.so.1.1: undefined symbol: EVP_idea_cbc, version OPENSSL_1_1_0
```

### Solution

Make sure you have OpenSSL 1.1 installed in your conda environment. You can install it using the following command:

```
conda install openssl==1.1
```
