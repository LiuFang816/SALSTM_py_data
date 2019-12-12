from django.conf.urls import url,include
import views

urlpatterns = [
    url(r'^$', views.index, name='app'),
    url(r'^status$', views.status, name='status'),
    url(r'^youtube$', views.yt, name='youtube'),
    url(r'^videos/$', views.VideoList.as_view()),
    url(r'^queries/$', views.QueryList.as_view()),
    url(r'^Search$', views.search),
    url(r'^videos/(?P<pk>\d+)/$', views.VideoDetail.as_view(), name='video_detail'),
    url(r'^frames/$', views.FrameList.as_view()),
    url(r'^frames/(?P<pk>\d+)/$', views.FrameDetail.as_view(), name='frames_detail'),
    url(r'^queries/(?P<pk>\d+)/$', views.QueryDetail.as_view(), name='query_detail'),

]
