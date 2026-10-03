from mathlib_mcp.extract.docstrings import clean_rst, parse_docstring

GOOGLE = """Solves a linear matrix equation.

It computes the exact solution of ``x`` in ``ax = b``.

Args:
    a (cupy.ndarray): The matrix with dimension ``(..., M, M)``.
    b (cupy.ndarray): The matrix with dimension ``(M,)`` or
        ``(..., M, K)``.

Returns:
    cupy.ndarray: The solution.

.. warning::
    May yield invalid results; see :func:`cupyx.errstate`.

.. seealso:: :func:`numpy.linalg.solve`
"""

NUMPY = """Compute something.

Parameters
----------
x : array_like
    Input values.
axis : int, optional
    Axis to use.

Returns
-------
out : ndarray
    Result.

Examples
--------
>>> f(1)
2
"""


def test_google_style_sections():
    doc = parse_docstring(GOOGLE)
    assert doc.summary == "Solves a linear matrix equation."
    assert doc.description == "It computes the exact solution of `x` in `ax = b`."
    assert doc.params["a"] == "(cupy.ndarray) The matrix with dimension `(..., M, M)`."
    assert doc.params["b"].endswith("`(M,)` or `(..., M, K)`.")
    assert doc.returns == "cupy.ndarray: The solution."
    assert "`cupyx.errstate`" in doc.warnings
    assert doc.see_also == ["numpy.linalg.solve"]


def test_numpy_style_sections_and_doctest():
    doc = parse_docstring(NUMPY)
    assert set(doc.params) == {"x", "axis"}
    assert doc.params["x"] == "array_like Input values."
    assert doc.returns.startswith("out : ndarray")
    assert doc.examples == ["f(1)"]


def test_signature_line_and_roles_are_cleaned():
    doc = parse_docstring("matmul(x1, x2, /, out=None, \\*\\*kwargs)\n\nMatrix product.")
    assert doc.summary == "Matrix product."
    assert clean_rst(":class:`~cupy.ndarray` and :math:`a @ b`") == "`cupy.ndarray` and `a @ b`"
