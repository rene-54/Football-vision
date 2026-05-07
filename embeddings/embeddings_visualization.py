import plotly.graph_objects as go
import numpy as np
from typing import Dict,List
from IPython.core.display import display, HTML
from PIL import Image
import base64
from io import BytesIO


def pil_image_to_data_uri(image: Image.Image) -> str:
    # Convert a PIL image into a base64 string so it can be embedded directly into HTML
    buffered = BytesIO()
    image.save(buffered, format="PNG")  # save image into memory instead of a file
    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")  # encode to base64 string
    return f"data:image/png;base64,{img_str}"  # return in browser-readable format


def display_projections(
    labels: np.ndarray,
    projections: np.ndarray,
    images: List[Image.Image],
    show_legend: bool = False,
    show_markers_with_text: bool = True
) -> None:
    # Create a dictionary mapping image IDs to their base64 representations
    # This allows JavaScript to dynamically display images on click
    image_data_uris = {
        f"image_{i}": pil_image_to_data_uri(image)
        for i, image in enumerate(images)
    }

    # Create an array of image IDs to link each point in the plot to an image
    image_ids = np.array([f"image_{i}" for i in range(len(images))])

    unique_labels = np.unique(labels)  # find all distinct classes
    traces = []

    # Create a separate 3D scatter trace for each label/class
    for unique_label in unique_labels:
        mask = labels == unique_label  # filter points belonging to this class

        trace = go.Scatter3d(
            x=projections[mask][:, 0],  # X coordinates
            y=projections[mask][:, 1],  # Y coordinates
            z=projections[mask][:, 2],  # Z coordinates
            mode='markers+text' if show_markers_with_text else 'markers',  # optionally show labels
            text=labels[mask],  # text shown on hover or next to markers
            customdata=image_ids[mask],  # attach image IDs for click interaction
            name=str(unique_label),  # label name in legend
            marker=dict(size=8),
            hovertemplate="<b>class: %{text}</b><br>image ID: %{customdata}<extra></extra>"  # custom hover info
        )

        traces.append(trace)

    # Compute global axis range so all axes use the same scale (keeps cube shape)
    all_axes = projections
    min_val = np.min(all_axes)
    max_val = np.max(all_axes)
    padding = (max_val - min_val) * 0.05  # add small margin for better visualization
    axis_range = [min_val - padding, max_val + padding]

    fig = go.Figure(data=traces)

    # Configure 3D scene layout
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=axis_range),
            yaxis=dict(title='Y', range=axis_range),
            zaxis=dict(title='Z', range=axis_range),
            aspectmode='cube'  # ensures equal scaling on all axes
        ),
        width=1000,
        height=1000,
        showlegend=show_legend
    )

    # Convert Plotly figure into an HTML div (without full page wrapper)
    plotly_div = fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        div_id="scatter-plot-3d"
    )

    # JavaScript handles click events on the plot
    # When a point is clicked, it retrieves the image ID and displays the corresponding image
    javascript_code = f"""
    <script>
        function displayImage(imageId) {{
            var imageElement = document.getElementById('image-display');
            var placeholderText = document.getElementById('placeholder-text');
            var imageDataURIs = {image_data_uris};

            imageElement.src = imageDataURIs[imageId]; // set selected image
            imageElement.style.display = 'block';
            placeholderText.style.display = 'none'; // hide placeholder text
        }}

        var chartElement = document.getElementById('scatter-plot-3d');
        chartElement.on('plotly_click', function(data) {{
            var customdata = data.points[0].customdata; // retrieve image ID
            displayImage(customdata);
        }});
    </script>
    """

    # Full HTML layout combining the plot and image display panel
    html_template = f"""
    <!DOCTYPE html>
    <html>
        <head>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                #image-container {{
                    position: fixed;
                    top: 0; left: 0;
                    width: 200px; height: 200px;
                    padding: 5px;
                    border: 1px solid #ccc;
                    background-color: white;
                    z-index: 1000;
                    box-sizing: border-box;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    text-align: center;
                }}

                #image-display {{
                    width: 100%; height: 100%;
                    object-fit: contain; // keeps image aspect ratio
                }}
            </style>
        </head>

        <body>
            {plotly_div}

            <!-- Floating container that shows the clicked image -->
            <div id="image-container">
                <img id="image-display" src="" alt="Selected image" style="display: none;" />
                <p id="placeholder-text">Click on a data entry to display an image</p>
            </div>

            {javascript_code}
        </body>
    </html>
    """

    # Render the final interactive HTML in the notebook
    display(HTML(html_template))