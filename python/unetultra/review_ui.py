"""Notebook widgets for mask review (01_masks.ipynb). Needs ipywidgets, matplotlib and ipympl."""

import io

import numpy as np
from PIL import Image

from .masks import overlay

STATUS_COLOURS = {"pending": "#b58900", "accepted": "#2a9d3a", "rejected": "#d62828"}


def _thumbnail_png(array, width):
    picture = Image.fromarray(array)
    picture.thumbnail((width, width))
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG")
    return buffer.getvalue()


def _item_overlay(review, item):
    taps = [obj["tap_point_image"] for obj in item.meta["objects"]]
    extra = [p for obj in item.meta["objects"] for p in obj.get("extra_points", [])]
    return overlay(review.image(item), item.masks, taps, extra)


def review_grid(review, page=0, per_page=12, columns=4, thumb_width=320, only=None):
    """A page of mask overlays, each with accept / reject buttons. Returns an ipywidgets box.

    `only` filters by status, e.g. only="pending". Remember to call review.save() afterwards."""
    import ipywidgets as widgets

    items = [item for item in review.items if only is None or item.status == only]
    cells = []
    for item in items[page * per_page : (page + 1) * per_page]:
        picture = widgets.Image(value=_thumbnail_png(_item_overlay(review, item), thumb_width), format="png")
        title = widgets.HTML()
        buttons = widgets.ToggleButtons(options=["pending", "accepted", "rejected"], value=item.status,
                                        style={"button_width": "76px"})

        def refresh(item=item, title=title):
            n = len(item.meta["objects"]) or "no"
            title.value = (f"<b>{item.sample_id}</b> · {n} object(s) · "
                           f"<span style='color:{STATUS_COLOURS[item.status]}'>{item.status}</span>")

        def on_change(change, item=item, refresh=refresh):
            review.set_status(item.key, change["new"])
            refresh()

        buttons.observe(on_change, names="value")
        refresh()
        cells.append(widgets.VBox([title, picture, buttons], layout=widgets.Layout(margin="4px")))
    pages = max(1, -(-len(items) // per_page))
    header = widgets.HTML(f"Page {page + 1} of {pages} · {len(items)} sample(s) · {review.counts()}")
    grid = widgets.GridBox(cells, layout=widgets.Layout(grid_template_columns=f"repeat({columns}, {thumb_width + 16}px)"))
    return widgets.VBox([header, grid])


def fix_mask(review, key, object_index=0, figsize=(10, 7.5)):
    """Interactive fix-up for one object: click to add a point, SAM re-runs, the overlay updates.

    Run `%matplotlib widget` first. Choose positive or negative with the toggle, then click on
    the image. Returns an ipywidgets box."""
    import ipywidgets as widgets
    import matplotlib.pyplot as plt

    item = review[key]
    kind = widgets.ToggleButtons(options=[("add to mask", True), ("remove from mask", False)])
    clear = widgets.Button(description="clear clicks")
    accept = widgets.Button(description="accept", button_style="success")
    reject = widgets.Button(description="reject", button_style="danger")
    status = widgets.HTML()
    with plt.ioff():
        figure, axis = plt.subplots(figsize=figsize)
    figure.canvas.header_visible = False
    artist = axis.imshow(_item_overlay(review, item))
    axis.set_axis_off()
    figure.tight_layout()

    def refresh():
        artist.set_data(_item_overlay(review, item))
        area = int(np.sum(item.masks[object_index])) if item.masks else 0
        status.value = f"{item.sample_id} · object {object_index + 1} · {area} px · {item.status}"
        figure.canvas.draw_idle()

    def on_click(event):
        if event.inaxes is axis and event.xdata is not None:
            review.add_click(key, object_index, event.xdata, event.ydata, positive=kind.value)
            refresh()

    def on_clear(_):
        review.clear_clicks(key, object_index)
        refresh()

    figure.canvas.mpl_connect("button_press_event", on_click)
    clear.on_click(on_clear)
    accept.on_click(lambda _: (review.set_status(key, "accepted"), refresh()))
    reject.on_click(lambda _: (review.set_status(key, "rejected"), refresh()))
    refresh()
    return widgets.VBox([widgets.HBox([kind, clear, accept, reject]), status, figure.canvas])
