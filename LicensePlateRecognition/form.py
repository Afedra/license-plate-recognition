from django import forms
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from django.conf import settings

def SignupDomainValidator(value):
    if '*' not in settings.ALLOWED_EMAILS:
        try:
            domain = value[value.index("@"):]
            if domain not in settings.ALLOWED_EMAILS:
                raise ValidationError('Invalid domain. Allowed domains on this network: {0}'.format(settings.ALLOWED_EMAILS))  # noqa: E501

        except Exception:
            raise ValidationError('Invalid domain. Allowed domains on this network: {0}'.format(settings.ALLOWED_EMAILS))  # noqa: E501

def ExistingEmailValidator(value):
    if not User.objects.filter(email__iexact=value).exists():
        raise ValidationError('User with this Email doesn\'t exists.')

class SignInForm(forms.Form):
    email = forms.CharField(
    widget=forms.EmailInput(attrs={'class': 'form-control', 'placeholder': 'Email'}),
    required=True,
    max_length=75)
    password = forms.CharField(
        widget= forms.PasswordInput(attrs={'class': 'form-control','placeholder': 'Password'}),
        required=True)
    remember = forms.BooleanField(required=False,widget=forms.CheckboxInput())

    def __init__(self, *args, **kwargs):
        super(SignInForm, self).__init__(*args, **kwargs)
        self.fields['email'].validators.append(ExistingEmailValidator)
        self.fields['email'].validators.append(SignupDomainValidator)

    def clean(self):
        super(SignInForm, self).clean()
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')
        if not email:
            self._errors['email'] = self.error_class(['Email required'])
        password = self.cleaned_data.get('password')
        if not password:
            self._errors['password'] = self.error_class(['Passwords required'])
        return self.cleaned_data
