// Variables
let DATA_PATH = "..."

let parse_date = d3.timeParse("%y/%m/%d")
let format_date = d3.timeFormat("%Y-%m-%d")

let width = 600
    height = 400
    margin = {top: 80, right: 20, bottom: 20, left: 50}

let svg = d3.select("#figure")
            .append("svg")
                .attr("height", height)
                .attr("width", width)

let xScale = d3.scaleLinear()
               .range([0, width])

let yScale = d3.scaleLinear()
               .range([0, height])

let COLORS = ['rgba(166,206,227,.2)','rgba(31,120,180,.2)','rgba(178,223,138,.2)','rgba(51,160,44,.2)','rgba(251,154,153,.2)','rgba(227,26,28,.2)','rgba(253,191,111,.2)','rgba(255,127,0,.2)','rgba(202,178,214,.2)','rgba(106,61,154,.2)','rgba(255,255,153,.2)','rgba(177,89,40,.2)']
let color = d3.scaleOrdinal()
              .range(COLORS)

let TOOLTIP_COLORS = ['rgba(166,206,227,.8)','rgba(31,120,180,.8)','rgba(178,223,138,.8)','rgba(51,160,44,.8)','rgba(251,154,153,.8)','rgba(227,26,28,.8)','rgba(253,191,111,.8)','rgba(255,127,0,.8)','rgba(202,178,214,.8)','rgba(106,61,154,.8)','rgba(255,255,153,.8)','rgba(177,89,40,.8)']
let tooltip_colors = d3.scaleOrdinal()
                       .range(TOOLTIP_COLORS)


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

d3.select("#container") // Add other stuff as needed
  .append("g")
    .attr("id", "REPLACE")


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
d3.dsv("\t", DATA_PATH, function(d){
    return {
        // parse data
    }
}).then(function(data){
    color.domain(Array.from(new Set(data.map(d => d.REPLACE_WITH_COLOR_MAPPING))))
    tooltip_colors.domain(Array.from(new Set(data.map(d => d.REPLACE_WITH_COLOR_MAPPING))))

    let options = [REPLACE_WITH_LIST_FOR_DROPDOWN]
    
    d3.select("#dropdown")
      .selectAll('option')
      .data(options)
      .enter()
      .append('option')
      .text(d => d)
      .attr("value", d => d)

    d3.select("#dropdown")
      .on("change", function(d){
        let selectedValue = d3.select(this).property("value")
        let figureData = data.filter(d => d[SelectedValue] !== null) 
        createFigure(figureData, SelectedValue)
      })

    // Create Default
    let figureData = data.filter(d => d[REPLACE_WITH_DEFAULT_VALUE] !== null) 
    createFigure(figureData, REPLACE_WITH_DEFAULT_VALUE)
    
}).catch(function (error) {
    console.log(error)
})

function createFigure(figureData, SelectedValue){

    // Setup
    xScale.domain([d3.min(figureData.map(d => d.REPLACE_WITH_X_MAPPING)), d3.max(figureData.map(d => d.REPLACE_WITH_X_MAPPING))+.05])
    yScale.domain(d3.extent(figureData.map(d => d.REPLACE_WITH_Y_MAPPING)))

    // Enter
    d3.select("#main-title")
      .append("text")
        .attr("x", width/2)
        .attr("y", -margin.top+20)
        .text("Main Title")

    d3.select("#x-axis-label")
      .append("text")
        .attr("x", width/2)
        .attr("y", -margin.top+55)
        .text("X-AXIS")

    d3.select("#y-axis-label")
      .append("text")
        .attr("x", -height/2)
        .attr("y", -margin.left+15)
        .attr("transform", "rotate(-90)")
        .text("Y-axis")

    d3.select("#x-axis")
      .call(d3.axisBottom(xScale))

    d3.select("#y-axis")
      .call(d3.axisLeft(yScale))

    d3.selectAll("REPLACE_WITH_ITEM")
      .remove()

    d3.select("#something")
      .selectAll("something")
      .data(figureData)
      .enter()
      .append("something")
        .attr("cx", ...)
        .on("mouseover", function(event,d) {
            if (d[SelectedValue] !== 0) {
                tooltip.transition()
                       .duration(200)
                       .style("opacity", .9)
                       .style("background", tooltip_colors(d.region))
                tooltip.html(d["value1"] + "<br/>" + "value2: " + d["value2"] + "<br/>" + SelectedValue + ": " + d[SelectedValue])
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
}
