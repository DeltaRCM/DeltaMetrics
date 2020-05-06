************
Installing
************

.. note::

    There is no pypi or conda candidate for this package yet.
    

You can install the development version of the software by following the instructions below.

.. warning::
    
    DeltaMetrics is currently in development, with a constantly changing API.
    We make no guarantee of stability, correctness, or functionality.



Latest version install
----------------------

Follow these instructions to install the latest development version of DeltaMetrics.

.. note:: 
    Developers should follow the developer instructions in the :doc:`../guides/devguide`.

Install by cloning:

.. code:: console

    $ git clone https://github.com/DeltaRCM/DeltaMetrics.git

and then ``cd`` into the directory and install editable copy.

.. code:: console

    $ cd DeltaMetrics
    $ pip install -e .

Check that your installation worked by running the tests.

.. code:: console

    $ pip install pytest
    $ pytest
