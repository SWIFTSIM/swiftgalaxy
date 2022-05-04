#!/bin/bash

# Formats the code.
black `find . -name "*.py"`

# Check compliance.
flake8 `find . -name "*.py"`
