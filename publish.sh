username='wilcoln'
password='g5XfiasLxxpLNXS'
# rm -rf build dist *.egg-info
python setup.py sdist bdist_wheel
#python3 -m twine upload dist/*
python -m twine upload --repository-url https://pypi.org/legacy/ dist/* || rm -rf build dist *.egg-info
# rm -rf build dist *.egg-info