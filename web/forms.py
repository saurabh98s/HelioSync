"""
Forms

This module contains form classes for the web interface.
"""

from flask_wtf import FlaskForm
from wtforms import StringField, TextAreaField, PasswordField, BooleanField, SelectField, IntegerField
from wtforms.validators import DataRequired, Email, Length, EqualTo, Optional

class LoginForm(FlaskForm):
    """Form for user login."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    remember_me = BooleanField('Remember Me')

class OrganizationForm(FlaskForm):
    """Form for creating/editing organizations."""
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=100)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])

class ProjectForm(FlaskForm):
    """Form for creating/editing projects."""
    name = StringField('Name', validators=[DataRequired(), Length(min=2, max=120)])
    description = TextAreaField('Description', validators=[Optional(), Length(max=500)])
    dataset_name = StringField('Dataset Name', validators=[DataRequired(), Length(max=50)])
    framework = SelectField('Framework', choices=[('pytorch', 'PyTorch'), ('tensorflow', 'TensorFlow')])
    min_clients = IntegerField('Minimum Clients', validators=[DataRequired()])
    rounds = IntegerField('Training Rounds', validators=[DataRequired()]) 