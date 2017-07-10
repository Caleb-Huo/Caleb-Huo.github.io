//
// format date as dd-mmm-yy
// example: 12-Jan-99
//
function date_ddmmmyy(date)
{
  var d = date.getDate();
  var m = date.getMonth() + 1;
  var y = date.getYear();

  // handle different year values 
  // returned by IE and NS in 
  // the year 2000.
  if(y >= 2000)
  {
    y -= 2000;
  }
  if(y >= 100)
  {
    y -= 100;
  }

  // could use splitString() here 
  // but the following method is 
  // more compatible
  var mmm = 
    ( 1==m)?'Jan':( 2==m)?'Feb':(3==m)?'Mar':
    ( 4==m)?'Apr':( 5==m)?'May':(6==m)?'Jun':
    ( 7==m)?'Jul':( 8==m)?'Aug':(9==m)?'Sep':
    (10==m)?'Oct':(11==m)?'Nov':'Dec';

  return "" +
    (d<10?"0"+d:d) + "-" +
    mmm + "-" +
    (y<10?"0"+y:y);
}


//
// get last modified date of the 
// current document.
//
function date_lastmodified()
{
  var lmd = document.lastModified;
  var s   = "Unknown";
  var d1;

  // check if we have a valid date
  // before proceeding
  if(0 != (d1=Date.parse(lmd)))
  {
    s = "" + date_ddmmmyy(new Date(d1));
  }

  return s;
}

