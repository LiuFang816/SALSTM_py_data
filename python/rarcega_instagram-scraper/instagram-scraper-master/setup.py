from setuptools import setup, find_packages

setup(name='instagram-scraper',
      version='1.1.0',
      description=("instagram-scraper is a command-line application written in Python"
                    " that scrapes and downloads an instagram user\'s photos and videos. Use responsibly."),
      url='https://github.com/rarcega/instagram-scraper',
      download_url='https://github.com/rarcega/instagram-scraper/tarball/1.1.0',
      author='Richard Arcega',
      author_email='hello@richardarcega.com',
      license='Public domain',
      packages=find_packages(exclude=['tests']),
      install_requires=["requests>=1.0.4",
                        "futures==2.2.0",
                        "tqdm>=3.8.0",
                        "requests_mock",
                        "nose"],
      entry_points = {
        'console_scripts': ['instagram-scraper=instagram_scraper.app:main'],
      },
      test_suite='nose.collector',
      zip_safe=False,
      keywords=['instagram', 'scraper', 'download', 'media', 'photos', 'videos'])