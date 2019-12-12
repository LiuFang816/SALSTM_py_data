from django.contrib import admin
from .models import Video,Frame,Detection,TEvent,IndexEntries,QueryResults,Query,FrameLabel


@admin.register(Video)
class VideoAdmin(admin.ModelAdmin):
    pass


@admin.register(QueryResults)
class QueryResultsAdmin(admin.ModelAdmin):
    pass


@admin.register(Query)
class QueryAdmin(admin.ModelAdmin):
    pass


@admin.register(FrameLabel)
class FrameLabelAdmin(admin.ModelAdmin):
    pass


@admin.register(Frame)
class FrameAdmin(admin.ModelAdmin):
    pass


@admin.register(IndexEntries)
class IndexEntriesAdmin(admin.ModelAdmin):
    pass


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    pass


@admin.register(TEvent)
class TEventAdmin(admin.ModelAdmin):
    pass


