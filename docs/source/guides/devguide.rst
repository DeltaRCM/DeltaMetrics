***************
Developer Guide
***************

Guide to developers!



Installation
------------

You should use your local installation for development. 

First, fork the repository to your own account on GitHub.
Then, clone your fork to your local machine:

.. code:: console

    $ git clone https://github.com/<username>/DeltaMetrics.git

and then ``cd`` into the directory and install editable copy.

.. code:: console

    $ cd DeltaMetrics
    $ pip install -e .

Check that your installation worked by running the tests.

.. code:: console

    $ pip install pytest
    $ pytest

You may wish to add the upstream repository for easy ``push``/``pull``.

.. code:: console

    $ git remote add upstream https://github.com/DeltaRCM/DeltaMetrics.git


Developing
----------

We explicitly ignore a folder called ``temp`` at the root of the project directory.
You should place any development projects, jupyter notebooks, images, sample files, etc that you want to work with, but are not part of the package itself, here.
