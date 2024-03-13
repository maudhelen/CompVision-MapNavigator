# CompVision-MapNavigator
The purpose of the project is to build a program which will take some hand gestures to navigate in a computer. For the sake of the example we will use it to neviagte through Google Maps.
We chose to go forward with two different models to see the differences between them

Before starting either, remember to initialize your virtual environment and run `pip install requirements.txt`

1. ### **Media Pipe model**
  - To run this one `python3 app.py`
  - [Jump to guide](#mediapipe-model)

    
2. ### **Keras model**
  - To run the keras model `python3 keras_app.py`
  - Guide
**Hand Gestures vary for each model**

# MediaPipe Model

## Hand Gestures Guide

<table>
  <tr>
    <td>
      <img src="images/annotated_closed.jpg" alt="Closed Palm" style="width:200px;"/>
    </td>
    <td>
      Closed Palm will simulate a left mouse click, and hold it down as long as the palm is closed.
    </td>
  </tr>
  <tr>
    <td>
      <img src="images/annotated_open.jpg" alt="Open Palm" style="width:200px;"/>
    </td>
    <td>
      Open Palm will move the cursor.
    </td>
  </tr>
  <tr>
    <td>
      <img src="images/annotated_right.jpg" alt="Pointing Right" style="width:200px;"/>
      <img src="images/annotated_right2.jpg" alt="Pointing Right 2" style="width:200px;"/>
    </td>
    <td>
      Pointing Right will simulate scrolling to the right or zooming in.
    </td>
  </tr>
  <tr>
    <td>
      <img src="images/annotated_left.jpg" alt="Pointing Left" style="width:200px;"/>
      <img src="images/annotated_left2.jpg" alt="Pointing Left 2" style="width:200px;"/>
    </td>
    <td>
      Pointing Left will simulate scrolling to the left or zooming out.
    </td>
  </tr>
</table>


## Architecture


# Keras Model



