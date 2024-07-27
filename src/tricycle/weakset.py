from collections.abc import MutableSet
from typing import Any
from weakref import WeakValueDictionary


class WeakSet(MutableSet):
    """A Set that uses weak references and does not check for equality.

    Normal sets check that two elements are equal by comparing __hash__
    and __eq__. For tensors, __eq__ is slow so this class only checks
    __hash__. This is a bad idea normally but because we control the hash
    for Tensors, we can (hopefully) use it for gradient calculation

    To avoid circular dependencies, we implement this as a weak set so that
    the garbage collector can clean up objects that are referred to by this
    class

    Attributes:
        _dict: A WeakValueDictionary to store the weak references.

    """

    def __init__(self, *args, **kwargs):
        """Initializes the WeakSet.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        super().__init__(*args, **kwargs)
        self._dict = WeakValueDictionary()

    def __contains__(self, x: Any) -> bool:
        """Checks if an element is in the set.

        Args:
            x: The element to check.

        Returns:
            bool: True if the element is in the set, False otherwise.
        """
        return hash(x) in self._dict

    def __iter__(self):
        """Returns an iterator over the elements in the set.

        Returns:
            iterator: An iterator over the values in the WeakValueDictionary.
        """
        return self._dict.values()

    def __len__(self):
        """Returns the number of elements in the set.

        Returns:
            int: The number of elements in the set.
        """
        return len(self._dict)

    def add(self, x: Any):
        """Adds an element to the set.

        Args:
            x: The element to add to the set.
        """
        self._dict[hash(x)] = x

    def discard(self, x: Any):
        """Removes an element from the set if it is present.

        Args:
            x: The element to remove from the set.
        """
        if hash(x) in self._dict:
            del self._dict[hash(x)]
