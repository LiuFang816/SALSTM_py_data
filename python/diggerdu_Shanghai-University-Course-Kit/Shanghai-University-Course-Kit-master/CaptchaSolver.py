#coding=utf-8

import requests as rq
import json

def solve(ImageName):
  im = open(ImageName,"rb")
  r = rq.post("http://115.159.223.206/shu_captcha/xkSolver.php",\
     files={'captcha' : (ImageName,im,'image/jpeg')})
  data = json.loads(r.content)
  return data["result"]
