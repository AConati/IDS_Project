{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "de4c07f2",
   "metadata": {},
   "outputs": [],
   "source": [
    "import ipywidgets as widgets\n",
    "from ipywidgets import AppLayout, Button, Layout, HTML\n",
    "from ipywidgets import GridspecLayout"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "83072766",
   "metadata": {},
   "outputs": [],
   "source": [
    "# We might be something we can use to upload static/sample reviews\n",
    "\n",
    "#widgets.FileUpload(\n",
    "#    accept='',  # Accepted file extension e.g. '.txt', '.pdf', 'image/*', 'image/*,.pdf'\n",
    "#    multiple=False  # True to accept multiple files upload else False\n",
    "#)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "c7017380",
   "metadata": {},
   "outputs": [],
   "source": [
    "def create_expanded_button(description, button_style):\n",
    "    return Button(description=description, button_style=button_style, layout=Layout(height='auto', width='auto'))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "61b635ae",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "70161a928e8349229fa761f067945780",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "GridspecLayout(children=(Button(button_style='warning', description='Button 0 - 0', layout=Layout(grid_area='w…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "grid = GridspecLayout(4, 3)\n",
    "\n",
    "for i in range(4):\n",
    "    for j in range(3):\n",
    "        grid[i, j] = create_expanded_button('Button {} - {}'.format(i, j), 'warning')\n",
    "grid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "ab3f7ffb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a7146e4803e04460b76724f9aa4a6b45",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "GridspecLayout(children=(Button(button_style='info', description='Topicnizer', layout=Layout(grid_area='widget…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "grid = GridspecLayout(7, 6, height='500px')\n",
    "grid[0, 0:5] = create_expanded_button('Topicnizer', 'info')\n",
    "grid[1:6, 0:1] = create_expanded_button('Dropdowns', 'info')\n",
    "grid[1:6, 1:3] = create_expanded_button('Filtered Reviews', 'warning')\n",
    "grid[1:6, 3:5] = create_expanded_button('Visuals', 'danger')\n",
    "grid[6, 0:5] = create_expanded_button('Footer', 'info')\n",
    "grid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "cc3c7ddc",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "a7146e4803e04460b76724f9aa4a6b45",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "GridspecLayout(children=(Button(button_style='info', description='Topicnizer', layout=Layout(grid_area='widget…"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "grid[1:2, 0] = widgets.Dropdown(\n",
    "    options=['happy', 'live', 'music'],\n",
    "    value='happy',\n",
    "    description='Topic 1:',\n",
    "    disabled=False,\n",
    "    layout=Layout(width='98%', height='30px')\n",
    ")\n",
    "\n",
    "grid[2:3, 0] = widgets.Dropdown(\n",
    "    options=['hot', 'diet', 'meal'],\n",
    "    value='diet',\n",
    "    description='Topic 2:',\n",
    "    disabled=False,\n",
    "    layout=Layout(width='98%', height='30px')\n",
    ")\n",
    "\n",
    "grid[3:4, 0] = widgets.Dropdown(\n",
    "    options=['fun', 'finger', 'sauce'],\n",
    "    value='fun',\n",
    "    description='Topic 3:',\n",
    "    disabled=False,\n",
    "    layout=Layout(width='98%', height='30px')\n",
    ")\n",
    "\n",
    "grid[4:5, 0] = widgets.Dropdown(\n",
    "    options=['classy', 'cheese', 'cake'],\n",
    "    value='classy',\n",
    "    description='Topic 4:',\n",
    "    disabled=False,\n",
    "    layout=Layout(width='98%', height='30px')\n",
    ")\n",
    "\n",
    "#grid[1:2,1:3] = widgets.HTML(\n",
    "#    value=\"<div>Filtered Review 1</div>\",\n",
    "#    placeholder='Some HTML',\n",
    "#    layout=Layout(justify-content='center')\n",
    "#)\n",
    "\n",
    "\n",
    "grid"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b292ce9e",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
