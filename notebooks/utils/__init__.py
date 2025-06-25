from functools import partial
from typing import Callable

import altair as alt

title: Callable[[str], alt.Title] = partial(alt.Title, color="darkslategrey", fontSize=25)
subtitle: Callable[[str], alt.Title] = partial(alt.Title, color="darkgrey", fontSize=16)
