
import matplotlib.axes as mpl_axes

class Chart:
  def __init__(self, axis: mpl_axes.Axes, title: str) -> None:
    self.axis = axis
    self.axis.set_title(title)
  
  def render(self) -> None:
    self.axis.relim()
    self.axis.autoscale_view()
