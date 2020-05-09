var width = 258,
    height = 258,
    fill = d3.scale.category20();

// mouse event vars
var selected_node = null,
    selected_link = null,
    mousedown_link = null,
    mousedown_node = null,
    mouseup_node = null;

// init svg
var outer = d3.select("#input")
  .append("svg:svg")
    .attr("width", width)
    .attr("height", height)
    .attr("pointer-events", "all");

var vis = outer
  .append('svg:g')
    .call(d3.behavior.zoom().on("zoom", rescale))
    .on("mousedown.zoom", null)
    .on("touchstart.zoom", null)
    .on("touchmove.zoom", null)
    .on("touchend.zoom", null)
    .on("dblclick.zoom", null)
  .append('svg:g')
    .on("mousemove", mousemove)
    .on("mousedown", mousedown)
    .on("mouseup", mouseup);

vis.append('svg:rect')
    .attr('width', width)
    .attr('height', height)
    .attr('fill', 'white')
    .attr('stroke-width', '2')
    .attr('stroke', 'rgb(224,224,224)');

// init force layout
var force = d3.layout.force()
    .size([width, height])
    .nodes([{label: 'LivingRoom0', idx: 41}]) // initialize with a single node
    .linkDistance(50)
    .charge(-200)
    .on("tick", tick);


// line displayed when dragging new nodes
var drag_line = vis.append("line")
    .attr("class", "drag_line")
    .attr("x1", 0)
    .attr("y1", 0)
    .attr("x2", 0)
    .attr("y2", 0);

// get layout properties
var nodes = force.nodes(),
    links = force.links(),
    node = vis.selectAll(".node"),
    link = vis.selectAll(".link");

// link weight
var current_weight = 1;

// room info
const room_idx_map = {"Background0": 0, "Bath0": 1, "Bath1": 2, "Bath2": 3,
"Bath3": 4, "Bath4": 5, "Bath5": 6, "Bath6": 7, "Bath7": 8, "Bath8": 9,
"Bath9": 10, "Bedroom0": 11, "Bedroom1": 12, "Bedroom2": 13, "Bedroom3": 14,
"Bedroom4": 15, "Bedroom5": 16, "Bedroom6": 17, "Bedroom7": 18, "Bedroom8": 19,
"Bedroom9": 20, "Dining0": 21, "Dining1": 22, "Dining2": 23, "Dining3": 24,
"Dining4": 25, "Entry0": 26, "Entry1": 27, "Entry2": 28, "Entry3": 29, "Entry4": 30,
"Garage0": 31, "Garage1": 32, "Garage2": 33, "Garage3": 34, "Garage4": 35,
"Kitchen0": 36, "Kitchen1": 37, "Kitchen2": 38, "Kitchen3": 39, "Kitchen4": 40,
"LivingRoom0": 41, "LivingRoom1": 42, "LivingRoom2": 43, "LivingRoom3": 44,
"LivingRoom4": 45, "Other0": 46, "Other1": 47, "Other10": 48, "Other11": 49,
"Other12": 50, "Other13": 51, "Other2": 52, "Other3": 53, "Other4": 54,
"Other5": 55, "Other6": 56, "Other7": 57, "Other8": 58, "Other9": 59, "Outdoor0": 60,
"Outdoor1": 61, "Outdoor2": 62, "Outdoor3": 63, "Outdoor4": 64, "Outdoor5": 65,
"Outdoor6": 66, "Outdoor7": 67, "Outdoor8": 68, "Outdoor9": 69, "Storage0": 70,
"Storage1": 71, "Storage2": 72, "Storage3": 73, "Storage4": 74, "Storage5": 75,
"Storage6": 76, "Storage7": 77, "Storage8": 78, "Storage9": 79}
var current_room = "Bedroom";
var room_idx = {"LivingRoom": 1};

function get_room_idx() {
  i = 0;
  if (current_room in room_idx) {
    i = room_idx[current_room];
  }
  label = current_room + i.toString();
  if (label in room_idx_map) {
    idx = room_idx_map[label];
    room_idx[current_room] = i+1;
    return [idx, label];
  } else {
    return [null, null];
  }
}

// button effect
selected_room_button = "Bedroom_button";
function select_room_button(ele) {
  ele.classList.add("button_selected");
  document.getElementById(selected_room_button).classList.remove("button_selected");
  selected_room_button = ele.id;
}
selected_weight_button = "direct_button";
function select_weight_button(ele) {
  ele.classList.add("button_selected");
  document.getElementById(selected_weight_button).classList.remove("button_selected");
  selected_weight_button = ele.id;
}

// color
room_color_map={
  'LivingRoom':"rgb(255,127,0)", 
  'Bedroom':"rgb(166,206,227)",
  'Kitchen':"rgb(253,191,111)",
  'Dining':"rgb(31,120,180)",
  'Bath':"rgb(178,223,138)",
  'Storage':"rgb(51,160,44)",
  'Entry':"rgb(227,26,28)",
  'Garage':"rgb(251,154,153)",
  'Other':"rgb(202,178,214)",
  'Outdoor':"rgb(106,61,154)",
};


// add keyboard callback
d3.select(window)
    .on("keydown", keydown);

redraw();

// focus on svg
// vis.node().focus();

function mousedown() {
  if (!mousedown_node && !mousedown_link) {
    // allow panning if nothing is selected
    vis.call(d3.behavior.zoom().on("zoom"), rescale);
    return;
  }
}

function mousemove() {
  if (!mousedown_node) return;

  // update drag line
  drag_line
      .attr("x1", mousedown_node.x)
      .attr("y1", mousedown_node.y)
      .attr("x2", d3.svg.mouse(this)[0])
      .attr("y2", d3.svg.mouse(this)[1]);

}

function mouseup() {
  if (mousedown_node) {
    // hide drag line
    drag_line
      .attr("class", "drag_line_hidden")

    if (!mouseup_node) {
      if (nodes.length === 35) {
        alert("Cannot have more than 35 rooms!");
        redraw();
        return;
      }
      rtn = get_room_idx();
      if (rtn[0] === null) {
        alert("Room type exceed limit!")
        return
      }
      idx = rtn[0];
      label = rtn[1];
      // add node
      var point = d3.mouse(this),
        node = {x: point[0], y: point[1], label: label, idx: idx};
        n = nodes.push(node);

      // select new node
      selected_node = node;
      selected_link = null;
      
      // add link to mousedown node
      links.push({source: mousedown_node, target: node, weight: current_weight});
    }

    redraw();
  }
  // clear mouse event vars
  resetMouseVars();
}

function resetMouseVars() {
  mousedown_node = null;
  mouseup_node = null;
  mousedown_link = null;
}

function tick() {
  link.attr("x1", function(d) { return d.source.x; })
      .attr("y1", function(d) { return d.source.y; })
      .attr("x2", function(d) { return d.target.x; })
      .attr("y2", function(d) { return d.target.y; })


  node.attr("cx", function(d) { return d.x; })
      .attr("cy", function(d) { return d.y; });
}

// rescale g
function rescale() {
  trans=d3.event.translate;
  scale=d3.event.scale;

  vis.attr("transform",
      "translate(" + trans + ")"
      + " scale(" + scale + ")");
}

function get_dash(weight) {
  if (weight === 1) {
    return "4,0";
  }
  return "4,4";
}

// redraw force layout
function redraw() {

  link = link.data(links);

  link.enter().insert("line", ".node")
      .attr("class", "link")
      .style("stroke-dasharray", function(d) {
        return get_dash(d.weight);
      })
      .on("mousedown", 
        function(d) { 
          mousedown_link = d; 
          if (mousedown_link == selected_link) selected_link = null;
          else selected_link = mousedown_link; 
          selected_node = null; 
          redraw(); 
        });

  link.exit().remove();

  link
    .classed("link_selected", function(d) { return d === selected_link; });

  node = node.data(nodes);

  node.enter().insert("circle")
      .attr("class", "node")
      .attr("r", 5)
      .style("fill", function(d) {return room_color_map[d.label.slice(0, d.label.length-1)]})
      .on("mousedown", 
        function(d) { 
          // disable zoom
          vis.call(d3.behavior.zoom().on("zoom"), null);

          mousedown_node = d;
          if (mousedown_node == selected_node) selected_node = null;
          else selected_node = mousedown_node; 
          selected_link = null; 

          // reposition drag line
          drag_line
              .attr("class", "link")
              .attr("x1", mousedown_node.x)
              .attr("y1", mousedown_node.y)
              .attr("x2", mousedown_node.x)
              .attr("y2", mousedown_node.y);

          redraw(); 
        })
      .on("mousedrag",
        function(d) {
          // redraw();
        })
      .on("mouseup", 
        function(d) { 
          if (mousedown_node) {
            mouseup_node = d; 
            if (mouseup_node == mousedown_node) { resetMouseVars(); return; }

            // add link
            var link = {source: mousedown_node, target: mouseup_node, weight: current_weight};
            links.push(link);

            // select new link
            selected_link = link;
            selected_node = null;

            // enable zoom
            vis.call(d3.behavior.zoom().on("zoom"), rescale);
            redraw();
          } 
        })
    .transition()
      .duration(750)
      .ease("elastic")
      .attr("r", 6.5);

  node.append("text")
      .attr("dx", 12)
      .attr("dy", ".35em")
      .attr("name", current_room)
      .text(function(d) { return d.name; });

  node.exit().transition()
      .attr("r", 0)
    .remove();

  node
    .classed("node_selected", function(d) { return d === selected_node; });

  

  if (d3.event) {
    // prevent browser's default behavior
    d3.event.preventDefault();
  }

  force.start();

}

function spliceLinksForNode(node) {
  toSplice = links.filter(
    function(l) { 
      return (l.source === node) || (l.target === node); });
  toSplice.map(
    function(l) {
      links.splice(links.indexOf(l), 1); });
}

function keydown() {
  if (!selected_node && !selected_link) return;
  switch (d3.event.keyCode) {
    case 8: // backspace
    case 46: { // delete
      if (selected_node) {
        nodes.splice(nodes.indexOf(selected_node), 1);
        spliceLinksForNode(selected_node);
      }
      else if (selected_link) {
        links.splice(links.indexOf(selected_link), 1);
      }
      selected_link = null;
      selected_node = null;
      redraw();
      break;
    }
  }
}

function prepareData() {
    data = {
        "rooms": [],
        "triples": [],
    };
    links.forEach(l => {
        s = l.source.idx;
        t = l.target.idx;
        data.triples.push([s, l.weight, t]);
        if (!(s in data.rooms)) {
            data.rooms.push(s)
        }
        if (!(t in data.rooms)) {
            data.rooms.push(t)
        }
    });
    return data;
}

var xxxxx = null;

function sendData() {
    jsonObj = prepareData();
    var xhr = new XMLHttpRequest();
    var url = "/generate";
    xhr.open("POST", url, true);
    xhr.setRequestHeader("Content-Type", "application/json");
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
          img_datauri = JSON.parse(xhr.response)["output"];
          document.getElementById("output").setAttribute("src", img_datauri);
        }
    };
    var data = JSON.stringify(jsonObj);
    xhr.send(data);
}