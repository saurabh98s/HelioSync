"""
Authentication Forms

This module provides forms for user authentication and registration.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField, SubmitField, TextAreaField
from wtforms.validators import DataRequired, Email, EqualTo, ValidationError, Length

from web.models import User, Organization

class LoginForm(FlaskForm):
    """Form for user login."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')
    submit = SubmitField('Sign In')

class RegistrationForm(FlaskForm):
    """Form for user registration."""
    username = StringField('Username', validators=[
        DataRequired(),
        Length(min=3, max=80, message='Username must be between 3 and 80 characters.')
    ])
    email = StringField('Email', validators=[
        DataRequired(),
        Email(message='Please enter a valid email address.'),
        Length(max=120)
    ])
    password = PasswordField('Password', validators=[
        DataRequired(),
        Length(min=8, message='Password must be at least 8 characters long.')
    ])
    password2 = PasswordField('Confirm Password', validators=[
        DataRequired(),
        EqualTo('password', message='Passwords must match.')
    ])
    submit = SubmitField('Register')

    def validate_username(self, username):
        """Validate that the username is not already in use."""
        user = User.query.filter_by(username=username.data).first()
        if user is not None:
            raise ValidationError('This username is already taken. Please choose a different one.')

    def validate_email(self, email):
        """Validate that the email is not already in use."""
        user = User.query.filter_by(email=email.data).first()
        if user is not None:
            raise ValidationError('This email address is already registered. Please use a different one.')

class OrganizationForm(FlaskForm):
    """Form for organization creation."""
    name = StringField('Organization Name', validators=[
        DataRequired(),
        Length(min=3, max=120, message='Organization name must be between 3 and 120 characters.')
    ])
    description = TextAreaField('Description', validators=[
        Length(max=500, message='Description cannot exceed 500 characters.')
    ])
    submit = SubmitField('Create Organization')

    def validate_name(self, name):
        """Validate that the organization name is not already in use."""
        org = Organization.query.filter_by(name=name.data).first()
        if org is not None:
            raise ValidationError('This organization name is already taken. Please choose a different one.') 