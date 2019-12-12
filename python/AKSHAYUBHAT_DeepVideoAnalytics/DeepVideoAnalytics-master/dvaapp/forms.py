from django import forms

class UploadFileForm(forms.Form):
    name = forms.CharField()
    file = forms.FileField()


class YTVideoForm(forms.Form):
    name = forms.CharField()
    url = forms.CharField()

