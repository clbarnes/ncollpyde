.. highlight:: shell

============
Contributing
============

Contributions are welcome, and they are greatly appreciated! Every little bit
helps, and credit will always be given.

You can contribute in many ways:

Types of Contributions
----------------------

Report Bugs
~~~~~~~~~~~

Report bugs at https://github.com/clbarnes/ncollpyde/issues.

If you are reporting a bug, please include:

* Your operating system name and version.
* Any details about your local setup that might be helpful in troubleshooting.
* Detailed steps to reproduce the bug.

Fix Bugs
~~~~~~~~

Look through the GitHub issues for bugs. Anything tagged with "bug" and "help
wanted" is open to whoever wants to implement it.

Implement Features
~~~~~~~~~~~~~~~~~~

Look through the GitHub issues for features.

Write Documentation
~~~~~~~~~~~~~~~~~~~

ncollpyde could always use more documentation, whether as part of the
official ncollpyde docs, in docstrings, or even on the web in blog posts,
articles, and such.

Submit Feedback
~~~~~~~~~~~~~~~

The best way to send feedback is to file an issue at https://github.com/clbarnes/ncollpyde/issues.

If you are proposing a feature:

* Explain in detail how it would work.
* Keep the scope as narrow as possible, to make it easier to implement.
* Remember that this is a volunteer-driven project, and that contributions
  are welcome :)

Get Started!
------------

Ready to contribute? Here's how to set up `ncollpyde` for local development.

1. Fork the `ncollpyde` repo on GitHub.
2. Clone your fork locally::

    $ git clone git@github.com:your_name_here/ncollpyde.git

3. Create a virtualenv, install the dependencies, and build the project. You will also need a `rust toolchain <https://www.rust-lang.org/tools/install>`_::

    $ cd ncollpyde/
    $ python -m venv --prompt ncollpyde env
    $ pip install -r requirements.txt -r docs/requirements.txt
    $ maturin develop

4. Create a branch for local development::

    $ git checkout -b name-of-your-bugfix-or-feature

   Now you can make your changes locally.

5. When you're done making changes, format your changes, then lint and test them (you may need to run ``maturin develop`` again to rebuild the rust components)::

    $ make fmt
    $ make lint
    $ make test

6. Commit your changes and push your branch to GitHub::

    $ git add .
    $ git commit
    $ git push origin name-of-your-bugfix-or-feature

7. Submit a pull request through the GitHub website.

Pull Request Guidelines
-----------------------

Before you submit a pull request, check that it meets these guidelines:

1. The pull request should include tests.
2. If the pull request adds functionality, the docs should be updated. Put
   your new functionality into a function with a docstring, and add the
   feature to the list in README.rst.
3. The pull request should work for CPython versions `supported by recent numpy releases <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_

Tips
----

To run a subset of tests::

    $ pytest tests.test_ncollpyde

Deploying
---------

A reminder for the maintainers on how to deploy.
Make sure all your changes are committed and passing CI.
Then run::

    $ cargo release minor  # or whatever version bump
