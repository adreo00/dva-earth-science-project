let DATA_PATH = "./classification_results.csv"

let parse_date = d3.timeParse("%Y-%m-%d")
let format_date = d3.timeFormat("%Y-%m")
function randDay() {
  return Math.floor(Math.random() * 29);
}
function gen_date(Year, Month) {
  return parse_date(Year+"-"+Month+"-"+randDay())
}

let margin = {top: 100, right: 50, bottom: 40, left: 50}
    width = 900
    height = 550
    offset = 50
    
let svg = d3.select("body")
            .append("svg")
                .attr("height", height+margin.top+margin.bottom)
                .attr("width", width+margin.right+margin.left)

let xScale = d3.scaleTime()
               .range([offset, width])

let binaryScale = d3.scalePoint()
                    .range([(offset+width)*.2, (offset+width)*.8])
                    .domain([0, 1])
                    .align(.5)

let yScale = d3.scalePoint()
               .range([height*.8, 0])
               .align(.5)

let rScale = d3.scaleRadial()
               .range([2, 10])

let COLORS = ["rgba(166,206,227,.2)","rgba(31,120,180,.2)","rgba(178,223,138,.2)","rgba(51,160,44,.2)","rgba(251,154,153,.2)","rgba(227,26,28,.2)","rgba(253,191,111,.2)","rgba(255,127,0,.2)","rgba(202,178,214,.2)","rgba(106,61,154,.2)","rgba(255,255,153,.2)","rgba(177,89,40,.2)"]
let color = d3.scaleOrdinal()
              .range(COLORS)

let TOOLTIP_COLORS = ["rgba(166,206,227,.8)","rgba(31,120,180,.8)","rgba(178,223,138,.8)","rgba(51,160,44,.8)","rgba(251,154,153,.8)","rgba(227,26,28,.8)","rgba(253,191,111,.8)","rgba(255,127,0,.8)","rgba(202,178,214,.8)","rgba(106,61,154,.8)","rgba(255,255,153,.8)","rgba(177,89,40,.8)"]
let tooltip_colors = d3.scaleOrdinal()
                       .range(TOOLTIP_COLORS)

let BIN_COLORS = ["rgba(161,217,155,1)", "rgba(240,240,240,1)"]
let binColors = d3.scaleOrdinal()
                  .domain([1, 0])
                  .range(BIN_COLORS)

let accToggle = false

// DOM Structure
svg.append("g")
   .attr("id", "container")
   .attr("transform", "translate("+margin.left+", "+margin.top+")")

d3.select("#container")
  .append("g")
    .attr("id", "main-title")
    .classed("title", true)
    .classed("main", true)

d3.select("#container")
  .append("g")
    .attr("id", "x-axis-label")
    .classed("title", true)
    .classed("sub", true)

d3.select("#container")
  .append("g")
    .attr("id", "y-axis-label")
    .classed("title", true)
    .classed("sub", true)

d3.select("#container")
  .append("g")
    .attr("id", "x-axis")

d3.select("#container")
  .append("g")
    .attr("id", "y-axis")

d3.select("#container")
  .append("g")
    .attr("id", "circle")

// Tooltips
let tooltip = d3.select("body")
                .append("div")
                    .classed("tooltip", true)
                    .style("opacity", 0)

let legend_tooltip = d3.select("body")
                       .append("div")
                            .classed("legend_tooltip", true)
                            .style("opacity", 0)

// Main
d3.dsv(",", DATA_PATH, function(d){
    return {
        "EQ": d.EQ,
        "Date": gen_date(d.Year, d.Month),
        "Type": d.Type,
        "Duration": +d.Duration,
        "Cleaning": d.Cleaning,
        "Model": d.Model,
        "Prediction": d.Prediction,
        "Correct": +d["Correct Prediction"]
    }
}).then(function(data){
    color.domain(Array.from(new Set(data.map(d => d.Prediction))))
    tooltip_colors.domain(Array.from(new Set(data.map(d => d.Prediction))))

    let options = Array.from(new Set(data.map(d => d.Cleaning)))
    
    d3.select("#dropdown")
      .selectAll("option")
      .data(options)
      .enter()
      .append("option")
      .text(d => d)
      .attr("value", d => d)

    d3.select("#dropdown")
      .on("change", function(d){
        let selectedValue = d3.select(this).property("value")
        let figureData = data.filter(d => d["Cleaning"] == selectedValue)
        createFigure(figureData, selectedValue)
      })

    d3.select("#vizButton")
      .on("click", function(d){
        if (accToggle) {
          accToggle = false
        } else {
          accToggle = true
        }
        let selectedValue = d3.select("#dropdown").property("value")
        let figureData = data.filter(d => d["Cleaning"] == selectedValue)
        createFigure(figureData, selectedValue)
      })

    // Create Default
    let figureData = data.filter(d => d["Cleaning"] == "Raw") 
    createFigure(figureData, "Raw")

    d3.select("#main-title")
      .append("text")
        .attr("x", offset+width/2)
        .attr("y", -margin.top+20)
        .text("Earthquake Classification Performance")

    d3.select("#y-axis")
      .call(d3.axisLeft(yScale))
      .call(g => g.select(".domain").remove())
    
}).catch(function (error) {
    console.log(error)
})


function createFigure(figureData, SelectedValue){
    // Setup
    yScale.domain(Array.from(new Set(figureData.map(d => d.Model))))
    rScale.domain(d3.extent(figureData.map(d => d.Duration)))
    d3.selectAll(".update")
      .remove()
    
    // Enter
    if (accToggle) {
      d3.select("#x-axis-label")
        .append("text")
          .classed("update", true)
          .attr("x", offset+width/2)
          .attr("y", height)
          .text("Classification Accuracy")

      d3.select("#x-axis")
        .call(d3.axisBottom(binaryScale))
        .attr('transform', 'translate(0,'+(height-margin.bottom)+')')

       d3.selectAll("circle")
        .remove()

      d3.select("#circle")
        .selectAll("circle")
        .data(figureData)
        .enter()
        .append("circle")
          .classed("circle", true)
          .attr("cx", d => binaryScale(d.Correct))
          .attr("cy", d => yScale(d.Model))
          .attr("fill", d => binColors(d.Correct))
          .attr("r", d => rScale(d.Duration))
          .on("mouseover", function(event,d) {
              if (d[SelectedValue] !== 0) {
                  tooltip.transition()
                         .duration(200)
                         .style("opacity", .9)
                         .style("background", tooltip_colors(d.Prediction))
                  tooltip.html(format_date(d["Date"]) + "<br/>" + "True Type: " + d["Type"] + "<br/>" + "Predicted Type: " + d["Prediction"])
                         .style("left", (event.pageX) + "px")
                         .style("top", (event.pageY - 28) + "px")
              }
          })
          .on("mouseout", function(d) {
              if (rScale(d[SelectedValue]) !== 0) {
                  tooltip.transition()
                         .duration(500)
                         .style("opacity", 0)
              }
          })
        .exit().remove()

        let simulation = d3.forceSimulation(figureData)
                           .force("x", d3.forceX(d => binaryScale(d.Correct))
                                         .strength(.34))
                           .force("y", d3.forceY(d => yScale(d.Model))
                                         .strength(1))
                           .force("collide", d3.forceCollide(d => rScale(d.Duration)*.8))
                           .alphaDecay(0)
                           .alpha(0.3)
                           .on("tick", tick)
        function tick() {
            d3.selectAll("circle")
                .attr("cx", d => d.x)
                .attr("cy", d => d.y)
        }
        let init_decay = setTimeout(function () {simulation.alphaDecay(0.1)}, 500)
      } else {
        // Setup
        xScale.domain(d3.extent(figureData.map(d => d.Date)))

        d3.select("#x-axis-label")
          .append("text")
            .classed("update", true)
            .attr("x", offset+width/2)
            .attr("y", height)
            .text("Date of Earthquake")

        d3.select("#x-axis")
          .call(d3.axisBottom(xScale))
          .attr('transform', 'translate(0,'+(height-margin.bottom)+')')

        d3.selectAll("circle")
          .remove()

        d3.select("#circle")
          .selectAll("circle")
          .data(figureData)
          .enter()
          .append("circle")
            .classed("circle", true)
            .attr("cx", d => xScale(d.Date))
            .attr("cy", d => yScale(d.Model))
            .attr("fill", d => color(d.Prediction))
            .attr("r", d => rScale(d.Duration))
            .on("mouseover", function(event,d) {
                if (d[SelectedValue] !== 0) {
                    tooltip.transition()
                           .duration(200)
                           .style("opacity", .9)
                           .style("background", tooltip_colors(d.Prediction))
                    tooltip.html(format_date(d["Date"]) + "<br/>" + "True Type: " + d["Type"] + "<br/>" + "Predicted Type: " + d["Prediction"])
                           .style("left", (event.pageX) + "px")
                           .style("top", (event.pageY - 28) + "px")
                }
            })
            .on("mouseout", function(d) {
                if (rScale(d[SelectedValue]) !== 0) {
                    tooltip.transition()
                           .duration(500)
                           .style("opacity", 0)
                }
            })
            .exit().remove()

          let simulation = d3.forceSimulation(figureData)
                             .force("x", d3.forceX(d => xScale(d.Date))
                                           .strength(.5))
                             .force("y", d3.forceY(d => yScale(d.Model))
                                           .strength(1))
                             .force("collide", d3.forceCollide(d => rScale(d.Duration)))
                             .alphaDecay(0)
                             .alpha(0.3)
                             .on("tick", tick)
          function tick() {
              d3.selectAll("circle")
                  .attr("cx", d => d.x)
                  .attr("cy", d => d.y)
          }
          let init_decay = setTimeout(function () {simulation.alphaDecay(0.1)}, 500)
      }
}


