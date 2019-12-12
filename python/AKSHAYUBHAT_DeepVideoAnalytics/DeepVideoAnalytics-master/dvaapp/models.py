from __future__ import unicode_literals

from django.db import models
from django.contrib.auth.models import User


class Query(models.Model):
    created = models.DateTimeField('date created', auto_now_add=True)
    results = models.BooleanField(default=False)
    task_id = models.CharField(max_length=100,default="")
    results_metadata = models.TextField(default="")
    user = models.ForeignKey(User, null=True)


class Video(models.Model):
    name = models.CharField(max_length=100,default="")
    length_in_seconds = models.IntegerField(default=0)
    height = models.IntegerField(default=0)
    width = models.IntegerField(default=0)
    metadata = models.TextField(default="")
    frames = models.IntegerField(default=0)
    created = models.DateTimeField('date created', auto_now_add=True)
    description = models.TextField(default="")
    uploaded = models.BooleanField(default=False)
    dataset = models.BooleanField(default=False)
    uploader = models.ForeignKey(User,null=True)
    detections = models.IntegerField(default=0)
    url = models.TextField(default="")
    youtube_video = models.BooleanField(default=False)
    query = models.BooleanField(default=False)
    parent_query = models.ForeignKey(Query,null=True)


class Frame(models.Model):
    video = models.ForeignKey(Video,null=True)
    frame_index = models.IntegerField()
    name = models.CharField(max_length=200,null=True)
    subdir = models.TextField(default="") # Retains information if the source is a dataset for labeling


class FrameLabel(models.Model):
    frame = models.ForeignKey(Frame)
    video = models.ForeignKey(Video)
    label = models.TextField()
    source = models.TextField()


class Detection(models.Model):
    video = models.ForeignKey(Video,null=True)
    frame = models.ForeignKey(Frame)
    object_name = models.CharField(max_length=100)
    confidence = models.FloatField(default=0.0)
    x = models.IntegerField(default=0)
    y = models.IntegerField(default=0)
    h = models.IntegerField(default=0)
    w = models.IntegerField(default=0)
    metadata = models.TextField(default="")


class IndexEntries(models.Model):
    video = models.ForeignKey(Video,null=True)
    framelist = models.CharField(max_length=100)
    algorithm = models.CharField(max_length=100)
    count = models.IntegerField()


class TEvent(models.Model):
    started = models.BooleanField(default=False)
    completed = models.BooleanField(default=False)
    video = models.ForeignKey(Video,null=True)
    operation = models.CharField(max_length=100,default="")
    created = models.DateTimeField('date created', auto_now_add=True)
    seconds = models.FloatField(default=-1)


class QueryResults(models.Model):
    query = models.ForeignKey(Query)
    video = models.ForeignKey(Video)
    frame = models.ForeignKey(Frame)
    rank = models.IntegerField()
    algorithm = models.CharField(max_length=100)
    distance = models.FloatField(default=0.0)