from django import forms


class ArticleAnalysisForm(forms.Form):
    title = forms.CharField(
        max_length=300, required=False,
        widget=forms.TextInput(attrs={"class": "form-control", "placeholder": "Article title (optional)"}),
    )
    content = forms.CharField(
        widget=forms.Textarea(attrs={"class": "form-control", "rows": 8,
                                      "placeholder": "Paste a news article to analyze..."}),
    )


class QueryForm(forms.Form):
    query = forms.CharField(
        max_length=300,
        widget=forms.TextInput(attrs={"class": "form-control",
                                       "placeholder": "e.g. Show me positive tech news from this week"}),
    )
