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


class StepReview:
    """Review masks one sample at a time: accept, reject, or edit, then move on.

    Editing is done by clicking on the image in one of four modes:
      add       click a part of the object the mask missed; SAM redraws
      remove    click something the mask wrongly covers; SAM redraws
      restart   click the object itself, for a tap that landed beside it; SAM starts again there
      outline   click points around the object, then "close outline": that shape becomes the mask

    The decision logic lives in plain methods (click, accept, reject, ...) so it can be tested
    without a notebook; widget() builds the controls around them.
    """

    MODES = ("add", "remove", "restart", "outline")

    def __init__(self, review, only="pending", autosave_every=10, log=print):
        self.review = review
        self.queue = [item for item in review.items
                      if item.meta["objects"] and (only is None or item.status == only)]
        self.position = 0
        self.object_index = 0
        self.mode = "add"
        self.zoom = True
        self.outline = []  # points of the outline being drawn
        self.autosave_every = autosave_every
        self.decisions = 0
        self.message = ""
        self._log = log
        self._refresh = lambda: None

    # -- state ------------------------------------------------------------------------------

    @property
    def item(self):
        return self.queue[self.position] if 0 <= self.position < len(self.queue) else None

    @property
    def finished(self):
        return self.position >= len(self.queue)

    def describe(self):
        if not self.queue:
            return "Nothing to review."
        if self.finished:
            return f"All {len(self.queue)} samples reviewed: {self.review.counts()}. Run the save cell."
        item = self.item
        obj = item.meta["objects"][self.object_index]
        distance = obj.get("lidar_distance_at_tap_m")
        where = f" · {distance * 100:.0f} cm" if distance else ""
        area = int(item.masks[self.object_index].sum()) if item.masks else 0
        objects = f" · object {self.object_index + 1}/{len(item.meta['objects'])}" if len(item.meta["objects"]) > 1 else ""
        return (f"{self.position + 1} of {len(self.queue)} · sample {item.sample_id}{objects}{where} · "
                f"mask {area} px · {item.status}" + (f" · {self.message}" if self.message else ""))

    def view_box(self):
        """(x0, x1, y0, y1) to display: the whole image, or a window around the object when zoomed."""
        item = self.item
        width, height = item.meta["image_size"]
        if not self.zoom:
            return 0, width, 0, height
        obj = item.meta["objects"][self.object_index]
        x, y = obj["tap_point_image"]
        expected = self.review.expected_area_px(obj, item.meta)
        half = max(0.06 * width, 2.5 * (expected ** 0.5 if expected else 0.1 * width))
        x0, x1 = max(0, x - half), min(width, x + half)
        y0, y1 = max(0, y - half), min(height, y + half)
        return x0, x1, y0, y1

    # -- actions ----------------------------------------------------------------------------

    def _step(self, delta):
        self.position = min(max(self.position + delta, 0), len(self.queue))
        self.object_index, self.outline, self.message = 0, [], ""
        self._refresh()

    def _decide(self, status):
        if self.finished:
            return
        self.review.set_status(self.item.key, status)
        self.decisions += 1
        if self.autosave_every and self.decisions % self.autosave_every == 0:
            self.review.save()
        self._step(+1)
        if self.finished:
            self.review.save()

    def accept(self):
        self._decide("accepted")

    def reject(self):
        self._decide("rejected")

    def skip(self):
        self._step(+1)

    def back(self):
        self._step(-1)

    def set_mode(self, mode):
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode, self.outline, self.message = mode, [], ""
        self._refresh()

    def click(self, x, y):
        """A click on the image at pixel (x, y), interpreted by the current mode."""
        if self.finished:
            return
        key = self.item.key
        if self.mode == "outline":
            self.outline.append([float(x), float(y)])
            self.message = f"outline: {len(self.outline)} point(s); close it when you have gone round"
        elif self.mode == "restart":
            self.review.move_tap(key, self.object_index, x, y)
            self.message = "restarted from your click"
        else:
            self.review.add_click(key, self.object_index, x, y, positive=self.mode == "add")
            self.message = ""
        self._refresh()

    def close_outline(self):
        if self.finished or len(self.outline) < 3:
            self.message = "an outline needs at least three points"
        else:
            self.review.set_polygon(self.item.key, self.object_index, self.outline)
            self.outline, self.message = [], "mask replaced by your outline"
        self._refresh()

    def other_guess(self):
        if self.finished:
            return
        position, count = self.review.next_candidate(self.item.key, self.object_index)
        self.message = f"SAM guess {position} of {count}"
        self._refresh()

    def reset(self):
        if self.finished:
            return
        self.review.clear_clicks(self.item.key, self.object_index)
        self.outline, self.message = [], "back to the mask from the tap alone"
        self._refresh()

    def next_object(self):
        if not self.finished:
            self.object_index = (self.object_index + 1) % len(self.item.meta["objects"])
            self.outline = []
            self._refresh()

    def save(self):
        counts = self.review.save()
        self.message = f"saved: {counts}"
        self._refresh()
        return counts

    # -- notebook UI ------------------------------------------------------------------------

    def widget(self, figsize=(9, 7)):
        """Build the review controls. Run `%matplotlib widget` first."""
        import ipywidgets as widgets
        import matplotlib.pyplot as plt

        status = widgets.HTML()
        modes = widgets.ToggleButtons(
            options=[("add to mask", "add"), ("remove from mask", "remove"),
                     ("restart from click", "restart"), ("draw outline", "outline")],
            value=self.mode, style={"button_width": "130px"})

        def button(label, action, style=""):
            control = widgets.Button(description=label, button_style=style, layout=widgets.Layout(width="auto"))
            control.on_click(lambda _: action())
            return control

        def toggle_zoom():
            self.zoom = not self.zoom
            self._refresh()

        decide = widgets.HBox([
            button("✓ accept & next", self.accept, "success"), button("✗ reject & next", self.reject, "danger"),
            button("skip", self.skip), button("back", self.back), button("save now", self.save, "info"),
        ])
        edit = widgets.HBox([
            button("other SAM guess", self.other_guess), button("close outline", self.close_outline),
            button("reset mask", self.reset), button("zoom in/out", toggle_zoom), button("next object", self.next_object),
        ])
        with plt.ioff():
            figure, axis = plt.subplots(figsize=figsize)
        figure.canvas.header_visible = False
        figure.canvas.toolbar_visible = False
        figure.subplots_adjust(0, 0, 1, 1)

        def refresh():
            status.value = f"<b>{self.describe()}</b>"
            axis.clear()
            axis.set_axis_off()
            item = self.item
            if item is not None:
                obj = item.meta["objects"][self.object_index]
                picture = overlay(self.review.image(item), item.masks, [obj["tap_point_image"]],
                                  obj.get("extra_points", []), alpha=0.45)
                axis.imshow(picture)
                if item.masks:
                    axis.contour(item.masks[self.object_index], levels=[0.5], colors="white", linewidths=1)
                if self.outline:
                    xs, ys = zip(*self.outline)
                    axis.plot(xs, ys, "-o", color="cyan", markersize=4, linewidth=1.5)
                x0, x1, y0, y1 = self.view_box()
                axis.set_xlim(x0, x1)
                axis.set_ylim(y1, y0)
            figure.canvas.draw_idle()

        def on_click(event):
            if event.inaxes is axis and event.xdata is not None and event.button == 1:
                self.click(event.xdata, event.ydata)

        modes.observe(lambda change: self.set_mode(change["new"]), names="value")
        figure.canvas.mpl_connect("button_press_event", on_click)
        self._refresh = refresh
        refresh()
        return widgets.VBox([status, decide, modes, edit, figure.canvas])


def step_review(review, only="pending", autosave_every=10):
    """One-sample-at-a-time mask review for a notebook. Returns (widget, stepper)."""
    stepper = StepReview(review, only=only, autosave_every=autosave_every)
    return stepper.widget(), stepper
