"""
Forms package for the Federated Learning web interface.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, SelectField
from wtforms.validators import DataRequired, Email, Length, EqualTo, ValidationError
from web.models import User, Organization

class LoginForm(FlaskForm):
    """Form for user login."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    """Form for user registration."""
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=64)])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    password2 = PasswordField('Repeat Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Register')

    def validate_username(self, username):
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('Please use a different username.')

    def validate_email(self, email):
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('Please use a different email address.')

class OrganizationForm(FlaskForm):
    """Form for organization creation and editing."""
    name = StringField('Organization Name', validators=[DataRequired(), Length(min=2, max=64)])
    description = StringField('Description', validators=[Length(max=256)])
    submit = SubmitField('Save Organization')

    def validate_name(self, name):
        org = Organization.query.filter_by(name=name.data).first()
        if org is not None:
            raise ValidationError('Please use a different organization name.')

class ApiKeyForm(FlaskForm):
    """Form for API key creation."""
    name = StringField('Key Name', validators=[DataRequired(), Length(min=2, max=64)])
    description = StringField('Description', validators=[Length(max=256)])
    submit = SubmitField('Create API Key') 